import sys
import re

filepath = '/home/hackbrian/Documents/gits/ReinFlow_IsaacSim/env/gym_utils/wrapper/xarm_isaac_env.py'
with open(filepath, 'r') as f:
    content = f.read()

# Make sure we import Float64MultiArray
if 'Float64MultiArray' not in content:
    content = re.sub(r'from geometry_msgs\.msg import PoseStamped',
                     r'from geometry_msgs.msg import PoseStamped\n        from std_msgs.msg import Float64MultiArray',
                     content)
    # Also for module level just in case
    content = re.sub(r'import gym', r'import gym\nfrom std_msgs.msg import Float64MultiArray\nimport time', content)


# Add publisher for gripper in _init_ros2
if 'self.isaac_gripper_pub' not in content:
    content = re.sub(r'self\.respawn_pub = self\.node\.create_publisher\(Twist, "/respawn", 1\)',
                     r'self.respawn_pub = self.node.create_publisher(Twist, "/respawn", 1)\n        self.isaac_gripper_pub = self.node.create_publisher(Float64MultiArray, "/position_controller/commands", 1)',
                     content)

# We need to implement the grasp sequence in step()
# Find where success is evaluated
grasp_logic = """
        success = (dist_xy < 0.02) and (ee_pos[2] <= self.trigger_z)

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
"""

content = re.sub(r'success = \(dist_xy < 0\.02\) and \(ee_pos\[2\] <= self\.trigger_z\)', grasp_logic, content)

# Also, when resetting the environment, we must open the gripper!
reset_gripper_logic = """
        if self.mode == "ros2_sync":
            from geometry_msgs.msg import Twist
            from std_msgs.msg import Float64MultiArray
            # Open gripper
            g_msg = Float64MultiArray()
            g_msg.data = [0.0]
            if hasattr(self, 'isaac_gripper_pub'):
                self.isaac_gripper_pub.publish(g_msg)
            
            twist = Twist()
"""

content = re.sub(r'if self\.mode == "ros2_sync":\n            from geometry_msgs\.msg import Twist\n            twist = Twist\(\)', reset_gripper_logic, content)


with open(filepath, 'w') as f:
    f.write(content)

