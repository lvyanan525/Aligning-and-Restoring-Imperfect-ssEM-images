import cv2
import torch
import numpy as np

def ensure_cv2_grayscale_format(np_arr):
    """Ensure numpy array is in cv2.imread(..., cv2.IMREAD_GRAYSCALE) format: (H, W), uint8, [0, 255]"""
    # Handle different input shapes
    if np_arr.ndim == 4:
        # (B, H, W, C) -> take first batch item -> (H, W, C)
        np_arr = np_arr[0]
    elif np_arr.ndim == 3:
        # (H, W, C) or (B, H, W) - check last dimension
        if np_arr.shape[-1] <= 4:  # Likely (H, W, C) format
            pass  # Keep as is
        else:
            # Likely (B, H, W) format, take first batch item
            np_arr = np_arr[0]
    
    # Now handle 3D arrays (H, W, C)
    if np_arr.ndim == 3:
        # (H, W, C) -> convert to grayscale
        if np_arr.shape[2] == 1:
            np_arr = np_arr[:, :, 0]  # (H, W, 1) -> (H, W)
        else:
            # Multi-channel: convert to grayscale (weighted average)
            np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGR2GRAY)
    elif np_arr.ndim == 2:
        # Already (H, W)
        pass
    else:
        raise ValueError(f"Unexpected array shape after processing: {np_arr.shape}")
    
    # Ensure uint8 dtype and [0, 255] range
    if np_arr.dtype != np.uint8:
        if np_arr.max() <= 1.0:
            np_arr = (np_arr * 255).astype(np.uint8)
        else:
            np_arr = np_arr.astype(np.uint8)
    
    return np_arr

def match_tensor_shape(tensor, target_shape):
    """Match tensor shape to target_shape, handling batch and channel dimensions"""
    if tensor.shape == target_shape:
        return tensor
    
    # Get target dimensions
    if len(target_shape) == 4:
        B_target, C_target, H_target, W_target = target_shape
    elif len(target_shape) == 3:
        C_target, H_target, W_target = target_shape
        B_target = None
    else:
        raise ValueError(f"Unexpected target shape: {target_shape}")
    
    # Get current tensor dimensions
    if len(tensor.shape) == 4:
        B_curr, C_curr, H_curr, W_curr = tensor.shape
    elif len(tensor.shape) == 3:
        C_curr, H_curr, W_curr = tensor.shape
        B_curr = None
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    # Resize spatial dimensions if needed
    if H_curr != H_target or W_curr != W_target:
        if B_curr is not None:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(H_target, W_target), 
                mode='bilinear', align_corners=False
            )
        else:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(H_target, W_target),
                mode='bilinear', align_corners=False
            ).squeeze(0)
    
    # Match batch dimension
    if B_target is not None:
        if B_curr is None:
            tensor = tensor.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            B_curr = 1
        
        if tensor.shape[0] != B_target:
            # Expand or repeat to match batch size
            if tensor.shape[0] == 1:
                # Use expand for efficient broadcasting (no memory copy)
                tensor = tensor.expand(B_target, -1, -1, -1)
            else:
                # Repeat if batch size doesn't match
                repeat_times = B_target // tensor.shape[0]
                remainder = B_target % tensor.shape[0]
                if remainder == 0:
                    tensor = tensor.repeat(repeat_times, 1, 1, 1)
                else:
                    tensor = torch.cat([
                        tensor.repeat(repeat_times, 1, 1, 1),
                        tensor[:remainder]
                    ], dim=0)
    
    # Match channel dimension
    if tensor.shape[1] != C_target:
        if tensor.shape[1] == 1:
            # Expand single channel to match target channels
            tensor = tensor.expand(-1, C_target, -1, -1)
        else:
            # Take first C_target channels or repeat
            if tensor.shape[1] > C_target:
                tensor = tensor[:, :C_target]
            else:
                # Repeat channels
                repeat_times = C_target // tensor.shape[1]
                remainder = C_target % tensor.shape[1]
                if remainder == 0:
                    tensor = tensor.repeat(1, repeat_times, 1, 1)
                else:
                    tensor = torch.cat([
                        tensor.repeat(1, repeat_times, 1, 1),
                        tensor[:, :remainder]
                    ], dim=1)
    
    return tensor

def match_mask_shape(tensor, target_img_shape):
    """Match mask tensor shape to [B, 1, H, W] format, matching batch and spatial dims from target_img_shape"""
    # Get target dimensions from image shape
    if len(target_img_shape) == 4:
        B_target, _, H_target, W_target = target_img_shape
    elif len(target_img_shape) == 3:
        _, H_target, W_target = target_img_shape
        B_target = None
    else:
        raise ValueError(f"Unexpected target image shape: {target_img_shape}")
    
    # Get current tensor dimensions
    if len(tensor.shape) == 4:
        B_curr, C_curr, H_curr, W_curr = tensor.shape
    elif len(tensor.shape) == 3:
        C_curr, H_curr, W_curr = tensor.shape
        B_curr = None
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    # Ensure single channel - take first channel if multi-channel
    if C_curr > 1:
        tensor = tensor[:, 0:1] if B_curr is not None else tensor[0:1]
        C_curr = 1
    
    # Resize spatial dimensions if needed
    if H_curr != H_target or W_curr != W_target:
        if B_curr is not None:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(H_target, W_target), 
                mode='bilinear', align_corners=False
            )
        else:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(H_target, W_target),
                mode='bilinear', align_corners=False
            ).squeeze(0)
    
    # Match batch dimension
    if B_target is not None:
        if B_curr is None:
            tensor = tensor.unsqueeze(0)  # (1, H, W) -> (1, 1, H, W)
            B_curr = 1
        
        if tensor.shape[0] != B_target:
            # Expand or repeat to match batch size
            if tensor.shape[0] == 1:
                # Use expand for efficient broadcasting (no memory copy)
                tensor = tensor.expand(B_target, -1, -1, -1)
            else:
                # Repeat if batch size doesn't match
                repeat_times = B_target // tensor.shape[0]
                remainder = B_target % tensor.shape[0]
                if remainder == 0:
                    tensor = tensor.repeat(repeat_times, 1, 1, 1)
                else:
                    tensor = torch.cat([
                        tensor.repeat(repeat_times, 1, 1, 1),
                        tensor[:remainder]
                    ], dim=0)
    
    # Ensure final shape is [B, 1, H, W]
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    
    return tensor