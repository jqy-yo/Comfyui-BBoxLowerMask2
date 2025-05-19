import numpy as np
import torch

class BBoxLowerMask2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "image_height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "bbox": ("LIST", {"default": [439.81, 252.41, 503.29, 337.41]}),
                "boundary": (["x1", "y1", "x2", "y2"], {"default": "y2"}),
                "offset": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    FUNCTION = "generate_mask"
    CATEGORY = "masking"
    DESCRIPTION = "Generates a binary mask by masking out one side of a bounding box edge (x1, y1, x2, or y2) with an optional offset. Useful for isolating lower, upper, left, or right regions relative to a detected box."

    def generate_mask(self, image_width, image_height, bbox, boundary, offset):
        # Extract the selected boundary value
        boundary_values = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
        threshold_value = boundary_values[boundary]
        
        # Apply offset
        threshold_value = int(threshold_value) + offset

        # Initialize mask with all 1s (white)
        mask_np = np.ones((image_height, image_width), dtype=np.float32)

        # Generate mask based on selected boundary and offset
        if boundary == "x1":
            mask_np[:, :threshold_value] = 0.0
        elif boundary == "y1":
            mask_np[:threshold_value, :] = 0.0
        elif boundary == "x2":
            mask_np[:, threshold_value:] = 0.0
        elif boundary == "y2":
            mask_np[threshold_value:, :] = 0.0

        # Convert to 1 x H x W tensor
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return (mask_tensor,)

NODE_CLASS_MAPPINGS = {
    "BBoxLowerMask2": BBoxLowerMask2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BBoxLowerMask2": "BBox Lower Mask 2",
}
