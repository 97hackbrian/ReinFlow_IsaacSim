# MIT License
#
# Copyright (c) 2025 ReinFlow Authors & robo_imitate contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Gymnasium environment for xArm Lite 6 - Pick Screwdriver in NVIDIA Isaac Sim.
Supports two modes:
1. 'standalone': Direct in-process execution via Isaac Sim SimulationApp (zero ROS overhead, fast).
2. 'ros2_sync': Synchronous lockstep bridge communicating with external Isaac Sim ROS 2 container.
"""

import os
import time
import math
import numpy as np
import gym
from std_msgs.msg import Float64MultiArray
import time
from gym import spaces


class XArmPickScrewdriverEnv(gym.Env):
    """
    RL Environment for ReinFlow interacting with the xArm Lite 6 screwdriver picking task.
    """
    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(
        self,
        mode="ros2_sync",
        usd_path=None,
        img_size=(96, 96),
        max_episode_steps=400,
        trigger_z=0.18,
        final_grasp_z=0.088,
        speed_multiplier=40.0,
        control_dt=0.05,
        sparse_reward=False,
        device="cuda:0",
        random_spawn=False,
        normalization_path=None,
        **kwargs
    ):
        super().__init__()
        self.mode = mode
        self.usd_path = usd_path
        self.img_size = img_size
        self.max_episode_steps = max_episode_steps
        self.trigger_z = trigger_z
        self.final_grasp_z = final_grasp_z
        self.speed_multiplier = speed_multiplier
        self.control_dt = control_dt
        self.sparse_reward = sparse_reward
        self.device = device
        self.random_spawn = random_spawn
        
        self.normalization_path = normalization_path
        self.obs_min, self.obs_max = None, None
        self.action_min, self.action_max = None, None
        if self.normalization_path is not None:
            data = np.load(self.normalization_path)
            self.obs_min = data['obs_min']
            self.obs_max = data['obs_max']
            self.action_min = data['action_min']
            self.action_max = data['action_max']

        # Bounds for object randomization (from robo_imitate)
        self.spawn_x_min, self.spawn_x_max = 0.22, 0.40
        self.spawn_y_min, self.spawn_y_max = -0.12, 0.18

        # Observation Space:
        # 'state': [x, y, z, roll, pitch, yaw] of the end-effector
        # 'rgb': [C, H, W] uint8 image
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "rgb": spaces.Box(low=0, high=255, shape=(3, img_size[0], img_size[1]), dtype=np.uint8)
        })

        # Action Space: [dx, dy, dz, droll, dpitch, dyaw]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Internal state tracking
        self.step_count = 0
        self.current_ee_pose = np.zeros(6, dtype=np.float32)
        self.target_spawn_x = 0.35
        self.target_spawn_y = 0.10
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.settled_steps = 0
        self.is_gripping = False

        if self.mode == "ros2_sync":
            self._init_ros2()
        elif self.mode == "standalone":
            self._init_standalone()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Expected 'ros2_sync' or 'standalone'.")

    def _init_ros2(self):
        """Initializes ROS 2 node and subscribers with synchronization locks."""
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Float64MultiArray
        from geometry_msgs.msg import Twist, Pose
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        import threading

        if not rclpy.ok():
            rclpy.init()

        self.node = Node("reinflow_isaac_bridge")
        self.bridge = CvBridge()
        self.obs_event = threading.Event()
        self.latest_rgb = np.zeros((3, self.img_size[0], self.img_size[1]), dtype=np.uint8)

        # Publishers
        self.target_raw_pub = self.node.create_publisher(PoseStamped, "/target_frame_raw", 1)
        self.respawn_pub = self.node.create_publisher(Twist, "/respawn", 1)
        self.isaac_gripper_pub = self.node.create_publisher(Float64MultiArray, "/position_controller/commands", 1)

        # Subscribers
        self.node.create_subscription(Image, "/rgb", self._ros2_img_callback, 1)
        self.node.create_subscription(PoseStamped, "/current_pose", self._ros2_pose_callback, 1)

        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

    def _ros2_img_callback(self, msg):
        import cv2
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if cv_img.shape[0] != self.img_size[0] or cv_img.shape[1] != self.img_size[1]:
            cv_img = cv2.resize(cv_img, (self.img_size[1], self.img_size[0]))
        # Transpose to (C, H, W)
        self.latest_rgb = np.transpose(cv_img, (2, 0, 1))
        self.obs_event.set()

    def _ros2_pose_callback(self, msg):
        import transforms3d as t3d
        pos = msg.pose.position
        ori = msg.pose.orientation
        euler = t3d.euler.quat2euler([ori.w, ori.x, ori.y, ori.z])
        self.current_ee_pose = np.array([pos.x, pos.y, pos.z, euler[0], euler[1], euler[2]], dtype=np.float32)

    def _init_standalone(self):
        """Initializes direct Isaac Sim Python Standalone API."""
        # This branch executes when ReinFlow runs directly with Isaac Sim python.sh
        # from omni.isaac.kit import SimulationApp
        # self.sim_app = SimulationApp({"headless": True, "open_usd": self.usd_path})
        pass

    def reset(self, **kwargs):
        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.settled_steps = 0
        self.is_gripping = False

        # Determine target position for screwdriver
        if self.random_spawn:
            self.target_spawn_x = np.random.uniform(self.spawn_x_min, self.spawn_x_max)
            self.target_spawn_y = np.random.uniform(self.spawn_y_min, self.spawn_y_max)
        else:
            self.target_spawn_x = 0.35
            self.target_spawn_y = 0.10

        
        if self.mode == "ros2_sync":
            from geometry_msgs.msg import Twist
            from std_msgs.msg import Float64MultiArray
            # Open gripper
            g_msg = Float64MultiArray()
            g_msg.data = [0.0]
            if hasattr(self, 'isaac_gripper_pub'):
                self.isaac_gripper_pub.publish(g_msg)
            
            twist = Twist()

            twist.linear.x = float(self.target_spawn_x)
            twist.linear.y = float(self.target_spawn_y)
            twist.linear.z = 0.012
            self.respawn_pub.publish(twist)
            # Wait for simulator tick
            self.obs_event.wait(timeout=2.0)

        obs = self._get_obs()
        return obs

    def normalize_obs(self, obs_state):
        if self.obs_min is not None:
            return 2 * ((obs_state - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)
        return obs_state
        
    def unnormalize_action(self, action):
        if self.action_min is not None:
            action = (action + 1) / 2.0
            return action * (self.action_max - self.action_min) + self.action_min
        return action

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)
        action = self.unnormalize_action(action)

        # Publish or execute action
        if self.mode == "ros2_sync":
            self.obs_event.clear()
            self._send_ros2_action(action)
            # Wait for next simulation step
            arrived = self.obs_event.wait(timeout=1.0)
            if not arrived:
                self.node.get_logger().warn("Isaac Sim observation timed out in step()")

        obs = self._get_obs()
        
        # Use RAW (unnormalized) ee pose for reward & success checks
        # obs["state"] is normalized for the policy, but reward needs real-world meters
        raw_ee_pos = self.current_ee_pose[:3]
        reward = self._compute_reward(raw_ee_pos, action)

        # Check termination & truncation using raw coordinates
        dist_xy = math.sqrt((raw_ee_pos[0] - self.target_spawn_x) ** 2 + (raw_ee_pos[1] - self.target_spawn_y) ** 2)
        
        success = (dist_xy < 0.02) and (raw_ee_pos[2] <= self.trigger_z)

        if success and self.mode == "ros2_sync":
            from geometry_msgs.msg import PoseStamped
            from std_msgs.msg import Float64MultiArray
            import time
            import transforms3d as t3d
            
            # Stop moving XY, drop to FINAL_GRASP_Z
            msg = PoseStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "link_base"
            
            # We must send absolute pose now since we are taking over control
            msg.pose.position.x = float(self.current_ee_pose[0])
            msg.pose.position.y = float(self.current_ee_pose[1])
            msg.pose.position.z = float(self.final_grasp_z)
            
            quat = t3d.euler.euler2quat(self.current_ee_pose[3], self.current_ee_pose[4], self.current_ee_pose[5])
            msg.pose.orientation.w = float(quat[0])
            msg.pose.orientation.x = float(quat[1])
            msg.pose.orientation.y = float(quat[2])
            msg.pose.orientation.z = float(quat[3])
            
            # Create absolute publisher if it doesn't exist (target_raw_pub is for relativ in gripper_link_base!)
            # Wait, no, target_frame_raw can take link_base if we change frame_id!
            # BUT cartesian_motion_controller might expect relative. Wait, speed limiter expects link_base!
            # Let's see _send_ros2_action. It uses "gripper_link_base".
            # We can just send a relative Z drop.
            # But the easiest is to publish to /target_frame_raw with link_base
            abs_pub = self.node.create_publisher(PoseStamped, '/target_frame_raw', 1)
            abs_pub.publish(msg)
            time.sleep(3.0) # wait to descend
            
            # Close gripper
            g_msg = Float64MultiArray()
            g_msg.data = [-0.01]
            self.isaac_gripper_pub.publish(g_msg)
            time.sleep(1.0)
            
            # Lift up
            msg.pose.position.z = 0.29
            abs_pub.publish(msg)
            time.sleep(1.5)
            
            # We don't reset the gripper here, it will be reset by the simulation when respawn happens.


        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_episode_steps)
        done = terminated or truncated

        info = {
            "success": float(success),
            "distance_xy": dist_xy,
            "ee_z": raw_ee_pos[2],
            "target_spawn": [self.target_spawn_x, self.target_spawn_y]
        }

        self.prev_action = action.copy()
        return obs, reward, done, False, info

    def _send_ros2_action(self, action):
        """Send action to Isaac Sim via ROS2.
        
        action is already unnormalized to physical units (meters/radians per step).
        action_min: [-0.064, -0.122, -0.265, -0.023, -0.037, -0.001] m
        action_max: [ 0.169,  0.175,  0.003,  0.016,  0.004,  0.011] m
        These are the actual delta displacements from the demonstration dataset.
        DO NOT multiply by speed_multiplier here - that would make values 40x too large.
        """
        from geometry_msgs.msg import PoseStamped
        import transforms3d as t3d

        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "gripper_link_base"

        # action is already in physical units — use directly
        msg.pose.position.x = float(action[0])
        msg.pose.position.y = float(action[1])
        msg.pose.position.z = float(action[2])

        quat = t3d.euler.euler2quat(action[3], action[4], action[5])
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])

        self.target_raw_pub.publish(msg)



    def _get_obs(self):
        return {
            "state": self.normalize_obs(self.current_ee_pose.copy()),
            "rgb": self.latest_rgb.copy()
        }

    def _compute_reward(self, raw_ee_pos, action):
        """Compute reward using raw (unnormalized) end-effector position in meters."""
        dist_xy = math.sqrt((raw_ee_pos[0] - self.target_spawn_x) ** 2 + (raw_ee_pos[1] - self.target_spawn_y) ** 2)
        
        if self.sparse_reward:
            if dist_xy < 0.02 and raw_ee_pos[2] <= self.trigger_z:
                return 10.0
            return 0.0

        # Dense kinematic reward
        # 1. Approach in XY
        r_reach = math.exp(- (dist_xy ** 2) / (2 * (0.05 ** 2)))

        # 2. Descent reward (only if settled in XY)
        r_descend = 0.0
        if dist_xy < 0.03:
            z_initial = 0.30
            r_descend = max(0.0, (z_initial - raw_ee_pos[2]) * 5.0)

        # 3. Trigger grasp bonus
        r_trigger = 0.0
        if dist_xy < 0.02 and raw_ee_pos[2] <= self.trigger_z:
            r_trigger = 5.0

        # 4. Action smoothness penalties
        r_penalty = 0.01 * np.sum(np.square(action)) + 0.005 * np.sum(np.square(action - self.prev_action))

        total_reward = r_reach + r_descend + r_trigger - r_penalty
        return float(total_reward)

    def render(self, mode="rgb_array", **kwargs):
        if mode == "rgb_array":
            # Return as (H, W, C) for rendering, we currently have (C, H, W)
            return np.transpose(self.latest_rgb, (1, 2, 0))
        return None
