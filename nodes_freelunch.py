#code originally taken from: https://github.com/ChenyangSi/FreeU (under MIT License)

import torch
import torch.fft as fft
import torch.nn.functional as F
import math

def normalize(latent, target_min=None, target_max=None):
    """
    Normalize a tensor `latent` between `target_min` and `target_max`.

    Args:
        latent (torch.Tensor): The input tensor to be normalized.
        target_min (float, optional): The minimum value after normalization. 
            - When `None` min will be tensor min range value.
        target_max (float, optional): The maximum value after normalization. 
            - When `None` max will be tensor max range value.

    Returns:
        torch.Tensor: The normalized tensor
    """
    min_val = latent.min()
    max_val = latent.max()
    
    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val
        
    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled

def hslerp(a, b, t):
    """
    Perform Hybrid Spherical Linear Interpolation (HSLERP) between two tensors.

    This function combines two input tensors `a` and `b` using HSLERP, which is a specialized
    interpolation method for smooth transitions between orientations or colors.

    Args:
        a (tensor): The first input tensor.
        b (tensor): The second input tensor.
        t (float): The blending factor, a value between 0 and 1 that controls the interpolation.

    Returns:
        tensor: The result of HSLERP interpolation between `a` and `b`.

    Note:
        HSLERP provides smooth transitions between orientations or colors, particularly useful
        in applications like image processing and 3D graphics.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")

    num_channels = a.size(1)
    
    interpolation_tensor = torch.zeros(1, num_channels, 1, 1, device=a.device, dtype=a.dtype)
    interpolation_tensor[0, 0, 0, 0] = 1.0

    result = (1 - t) * a + t * b

    if t < 0.5:
        result += (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    else:
        result -= (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor

    return result
    
def batched_slerp(a, b, t):
    # Ensure that tensors a and b have compatible shapes
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")

    # Compute the dot product between a and b along the appropriate dimension
    dot_product = torch.sum(a * b, dim=1)

    # Clamp the dot product to ensure it's within the valid range [-1, 1]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate the angle between the vectors using the dot product
    angle = torch.acos(dot_product)

    # Ensure that the angle is in the range [0, pi]
    angle = angle % (2 * torch.pi)

    # Compute the SLERP interpolation
    interpolated = (a * torch.sin((1 - t) * angle) + b * torch.sin(t * angle)) / torch.sin(angle)

    return interpolated

blending_modes = {

    # Args:
    #   - a (tensor): Latent input 1
    #   - b (tensor): Latent input 2
    #   - t (float): Blending factor

    # Interpolates between tensors a and b using normalized linear interpolation.
    'bislerp': lambda a, b, t: normalize((1 - t) * a + t * b),
    
    # Transfer the color from `b` to `a` by t` factor
    'colorize': lambda a, b, t: a + (b - a) * t,

    # Interpolates between tensors a and b using cosine interpolation.
    'cosine interp': lambda a, b, t: (a + b - (a - b) * torch.cos(t * torch.tensor(math.pi))) / 2,

    # Interpolates between tensors a and b using cubic interpolation.
    'cuberp': lambda a, b, t: a + (b - a) * (3 * t ** 2 - 2 * t ** 3),

    # Interpolates between tensors a and b using normalized linear interpolation,
    # with a twist when t is greater than or equal to 0.5.
    'hslerp': hslerp,

    # Adds tensor b to tensor a, scaled by t.
    'inject': lambda a, b, t: a + b * t,

    # Interpolates between tensors a and b using linear interpolation.
    'lerp': lambda a, b, t: (1 - t) * a + t * b,

    # Simulates a brightening effect by adding tensor b to tensor a, scaled by t.
    'linear dodge': lambda a, b, t: normalize(a + b * t),

    # Interpolates between tensors a and b using spherical linear interpolation (SLERP).
    'slerp': batched_slerp,
    
}

def Fourier_filter(x, threshold, scale, scales=None, strength=1.0):
    # FFT
    x_freq = fft.fftn(x.float(), dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale

    if scales is not None:
        for scale_params in scales:
            if isinstance(scale_params, tuple) and len(scale_params) == 2:
                scale_threshold, scale_value = scale_params
                # Apply strength to the scale_value
                scaled_scale_value = scale_value * strength
                scale_mask = torch.ones((B, C, H, W), device=x.device)
                scale_mask[..., crow - scale_threshold:crow + scale_threshold, ccol - scale_threshold:ccol + scale_threshold] = scaled_scale_value
                new_mask = mask * scale_mask

    # Blend the result with the original mask based on strength
    mask = mask + (new_mask - mask) * strength

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)

mscales = {

    "Default": None,

    "Bandpass": [
        (5, 0.0),    # Low-pass filter
        (15, 1.0),   # Pass-through filter (allows mid-range frequencies)
        (25, 0.0),   # High-pass filter
    ],

    "Low-Pass": [
        (10, 1.0),   # Allows low-frequency components, suppresses high-frequency components
    ],

    "High-Pass": [
        (10, 0.0),   # Suppresses low-frequency components, allows high-frequency components
    ],

    "Pass-Through": [
        (10, 1.0),   # Passes all frequencies unchanged, no filtering
    ],

    "Gaussian-Blur": [
        (10, 0.5),   # Blurs the image by allowing a range of frequencies with a Gaussian shape
    ],

    "Edge-Enhancement": [
        (10, 2.0),   # Enhances edges and high-frequency features while suppressing low-frequency details
    ],

    "Sharpen": [
        (10, 1.5),   # Increases the sharpness of the image by emphasizing high-frequency components
    ],

    "Multi-Bandpass": [
        [(5, 0.0), (15, 1.0), (25, 0.0)],  # Multi-scale bandpass filter
    ],

    "Multi-Low-Pass": [
        [(5, 1.0), (10, 0.5), (15, 0.2)],  # Multi-scale low-pass filter
    ],

    "Multi-High-Pass": [
        [(5, 0.0), (10, 0.5), (15, 0.8)],  # Multi-scale high-pass filter
    ],

    "Multi-Pass-Through": [
        [(5, 1.0), (10, 1.0), (15, 1.0)],  # Pass-through at different scales
    ],

    "Multi-Gaussian-Blur": [
        [(5, 0.5), (10, 0.8), (15, 0.2)],  # Multi-scale Gaussian blur
    ],

    "Multi-Edge-Enhancement": [
        [(5, 1.2), (10, 1.5), (15, 2.0)],  # Multi-scale edge enhancement
    ],

    "Multi-Sharpen": [
        [(5, 1.5), (10, 2.0), (15, 2.5)],  # Multi-scale sharpening
    ],

}

class WAS_FreeU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "multiscale_mode": (list(mscales.keys()),),
                    "multiscale_strength": ("FLOAT", {"default": 1.0, "max": 1.0, "min": 0, "step": 0.001}),
                    "b1": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "b2": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.001}),
                    "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.001}),
                },
                "optional": {
                    "b1_mode": (list(blending_modes.keys()),),
                    "b1_blend": ("FLOAT", {"default": 1.0, "max": 100, "min": 0, "step": 0.001}),
                    "b2_mode": (list(blending_modes.keys()),),
                    "b2_blend": ("FLOAT", {"default": 1.0, "max": 100, "min": 0, "step": 0.001}),
                    "threshold": ("FLOAT", {"default": 1.0, "max": 1.0, "min": 0, "step": 0.001}),
                    "override_scales": ("STRING", {"default": '''# Sharpen
# 10, 1.5''', "multiline": True}),
                }
        }
                            
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, multiscale_mode, multiscale_strength, b1, b2, s1, s2, b1_mode="add", b1_blend=1.0, b2_mode="add", b2_blend=1.0, threshold=1.0, override_scales=""):
        def output_block_patch(h, hsp, transformer_options):
            scales_list = []
            if override_scales.strip() != "":
                scales_str = override_scales.strip().splitlines()
                for line in scales_str:
                    if not line.strip().startswith('#') and not line.strip().startswith('!') and not line.strip().startswith('//'):
                        scale_values = line.split(',')
                        if len(scale_values) == 2:
                            scale_threshold, scale_value = int(scale_values[0]), float(scale_values[1])
                            scales_list.append((scale_threshold, scale_value))

            scales = mscales[multiscale_mode] if not scales_list else scales_list

            if h.shape[1] == 1280:
                h_t = h[:,:640]
                h_r = h_t * b1
                h[:,:640] = blending_modes[b1_mode](h_t, h_r, b1_blend)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s1, scales=scales, strength=multiscale_strength)
            if h.shape[1] == 640:
                h_t = h[:,:320]
                h_r = h[:,:320] * b2
                h[:,:320] = blending_modes[b2_mode](h_t, h_r, b2_blend)
                hsp = Fourier_filter(hsp, threshold=threshold, scale=s2, scales=scales, strength=multiscale_strength)
            return h, hsp

        m = model.clone()
        m.set_model_output_block_patch(output_block_patch)
        return (m, )


NODE_CLASS_MAPPINGS = {
    "FreeU (Advanced)": WAS_FreeU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeU (Advanced)": "FreeU (Advanced)",
}