import torch

if not hasattr(torch, "unravel_index"):
    def _unravel_index(indices, shape):
        if not isinstance(shape, (list, tuple)):
            shape = tuple(shape)
        indices = torch.as_tensor(indices)
        coords = []
        for dim in reversed(shape):
            coords.append(indices % dim)
            indices = torch.div(indices, dim, rounding_mode="floor")
        return tuple(reversed(coords))
    torch.unravel_index = _unravel_index
