# FreeU Advanced Plus
Let's say you and I grab dinner, and movie after lunch? 🌃📺😏
 
![image](https://github.com/WASasquatch/FreeU_Advanced/assets/1151589/c1dc2ec9-e6a3-4d2d-bf81-697e5d5aabcb)

### Exmaple of default node settings applied across blocks.
![default_block_examples](https://github.com/WASasquatch/FreeU_Advanced/assets/1151589/d01dea23-7ad6-4b89-ba43-70412afbd75f)


## Input Parameters

- `model` (`MODEL`): Model to patch
- `target_block` (`COMBO`): Which block to target; `input_block` or `output_block`
- `multiscale_mode` (`COMBO`): A list of available multiscale modes:
  - `["Default", "Bandpass", "Low-Pass", "High-Pass", "Pass-Through", "Gaussian-Blur", "Edge-Enhancement", "Sharpen", "Multi-Bandpass", "Multi-Low-Pass", "Multi-High-Pass", "Multi-Pass-Through", "Multi-Gaussian-Blur", "Multi-Edge-Enhancement", "Multi-Sharpen"]`
- `multiscale_strength` (`FLOAT`, Default: 1.0, Range: [0.0, 1.0], Step: 0.001): Strength of scaling
- `b1_slice` (`INT`, Default: 640, Range: [64, 1280], Step: 1): The size of the array slice for b1 operation
- `b2_slice` (`INT`, Default: 640, Range: [64, 640], Step: 1): The size of the array slice for b2 operation
- `b1` (`FLOAT`, Default: 1.1, Range: [0.0, 10.0], Step: 0.001): `b1`  output multiplier
- `b2` (`FLOAT`, Default: 1.2, Range: [0.0, 10.0], Step: 0.001): `b2`  output multiplier
- `s1` (`FLOAT`, Default: 0.9, Range: [0.0, 10.0], Step: 0.001): `s1` Fourier transform scale strength
- `s2` (`FLOAT`, Default: 0.2, Range: [0.0, 10.0], Step: 0.001): `s2` Fourier transform scale strength

### Optional Parameters

- `b1_mode` (`COMBO`): Blending modes for `b1` multiplied result.
  - `['bislerp', 'colorize', 'cosine interp', 'cuberp', 'hslerp', 'inject', 'lerp', 'linear dodge', 'slerp']`
- `b1_blend` (`FLOAT`, Default: 1.0, Range: [0.0, 100], Step: 0.001): Blending strength for `b1`.
- `b2_mode` (`COMBO`): Blending modes for `b2` multiplied result.
  - `['bislerp', 'colorize', 'cosine interp', 'cuberp', 'hslerp', 'inject', 'lerp', 'linear dodge', 'slerp']`
- `b2_blend` (`FLOAT`, Default: 1.0, Range: [0.0, 100], Step: 0.001): Blending strength for `b2`.
- `threshold` (`INT`, Default: 1.0, Range: [1, 10], Step: 1): The exposed threshold value of the Fourier transform function.
- `use_override_scales` (`COMBO`): "true", or "false" on whether to use `override_scales`
- `override_scales` (`STRING`, Default: [Multiline String]): Override scales. Create custom scales and experiment with results.
  - Example `10, 1.5` would create the `multiscale_mode` effect `Sharpen`
  - You can use `#`, `//` and `!` to comment out lines. 
