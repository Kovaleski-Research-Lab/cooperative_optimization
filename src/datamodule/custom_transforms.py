#--------------------------------
# Import: Python libraries
#--------------------------------

import torch
import torch.nn.functional as F
from loguru import logger

#--------------------------------
# Initialize: Wavefront transform
#--------------------------------

class WavefrontTransform(object):
    def __init__(self, params):
        self.params = params.copy()
        logger.debug("custom_transforms.py - Initializing WavefrontTransform")

        # Set initialization strategy for the wavefront
        self.phase_initialization_strategy = params['phase_initialization_strategy']

        if self.phase_initialization_strategy == 0:
            logger.debug("custom_transforms.py | WavefrontTransform | Phase Initialization : Phase = torch.ones(), Amplitude = Sample")
        else:
            logger.debug("custom_transforms.py | WavefrontTransform | Phase Initialization : Phase = Sample, Amplitude = torch.ones()")

    def __call__(self,sample):
        c,w,h = sample.shape 
        if self.phase_initialization_strategy == 0:
            phases = torch.ones(c,w,h)
            amplitude = sample
        else:
            phases = sample
            amplitude = torch.ones(c,w,h)

        return amplitude * torch.exp(1j*phases)

#--------------------------------
# Initialize: Normalize transform
#--------------------------------
class Normalize(object):                                                                    
    def __init__(self, params):                                                             
        self.params = params.copy()                                                         
        logger.debug("custom_transforms.py - Initializing Normalize")
                                                                                            
    def __call__(self,sample):                                                              
                                                                                            
        min_val = torch.min(sample)                                                         
        sample = sample - min_val                                                           
        max_val = torch.max(sample)                                                         
                                                                                            
        return sample / max_val 

#--------------------------------
# Initialize: Threshold transform
#--------------------------------

class Threshold(object):
    def __init__(self, threshold):
        logger.debug("custom_transforms.py - Initializing Threshold")
        self.threshold = threshold
        logger.debug("custom_transforms.py | Threshold | Setting threshold to {}".format(self.threshold))

    def __call__(self, sample):
        return (sample > self.threshold)


#--------------------------------
# Initialize: Linear shift transform
#--------------------------------

class LinearShift:
    def __init__(self, shift_x: int = 0, shift_y: int = 0):
        """
        Args:
            shift_x (int): Number of pixels to shift horizontally. Positive is right, negative is left.
            shift_y (int): Number of pixels to shift vertically. Positive is down, negative is up.
        """
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Ensure image is a tensor with a batch dimension
        if len(img.shape) == 2:  # If grayscale without channel dimension
            img = img.unsqueeze(0).unsqueeze(0)
        elif len(img.shape) == 3 and img.shape[0] == 1:  # If grayscale with channel dimension
            img = img.unsqueeze(0)
        
        # Original height and width
        _, _, h, w = img.size()
        
        # Padding needed for positive and negative shifts
        pad_left = max(0, -self.shift_x)
        pad_right = max(0, self.shift_x)
        pad_top = max(0, -self.shift_y)
        pad_bottom = max(0, self.shift_y)

        # Apply the padding
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))

        # Crop to the original size
        crop_top = pad_top
        crop_bottom = crop_top + h
        crop_left = pad_left
        crop_right = crop_left + w

        img = img[:, :, crop_top:crop_bottom, crop_left:crop_right]

        return img.squeeze(0) if len(img.shape) == 4 else img

