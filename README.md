# Diversity-Preserving Robust Watermarking (DPRW)

This repository provides the reference implementation of the paper *"Diversity-Preserving Robust Watermarking for Diffusion Model Generated Images"*. 




## Usage
The DPRW Engine provides three main functionalities:

* Generating Images with Watermarks: Embed a watermark into an image during generation.
* Extracting Watermarks: Invert an image to retrieve the embedded watermark.
* Batch Processing: Process a directory of images to extract and evaluate watermarks.

### Initialization

Create an instance of the DPRW_Engine class with your desired configuration:

```
from dprw_engine import DPRW_Engine

engine = DPRW_Engine(
    model_id="stabilityai/stable-diffusion-2-1-base",
    scheduler_type="DDIM",
    key_hex="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7",
    nonce_hex="05072fd1c2265f6f2e2a4080a2bfbdd8",
    use_seed=True,
    random_seed=42,
    seed_mode="sequential",
    device="cuda",
    dtype=torch.float32,
)
```

### Generating Images with Watermarks

Use the generate_image method to create an image with an embedded watermark:

```
engine.generate_image(
    prompt="Your text prompt here",
    width=512,
    height=512,
    num_steps=30,
    message="Your watermark message",
    message_length=256,
    window_size=1,
    batchsize=1,
    output_path="path/to/save/image.png"
)
```


### Extracting Watermarks

To extract a watermark, first invert the image to obtain the latent representation, then extract the watermark:


```
reversed_latents = engine.invert_image(
    image_path="path/to/image.png",
    width=512,
    height=512,
    num_steps=30,
    guidance_scale=7.5
)
extracted_bin, extracted_msg = engine.extract_watermark(
    reversed_latents,
    message_length=256,
    window_size=1
)

```



### Batch Processing

Process a directory of images to extract and evaluate watermarks:

```
engine.process_directory(
    dir_path="path/to/directory",
    message_length=256,
    window_size=1,
    threshold=0.7,
    original_msg="Your original watermark message",
    width=512,
    height=512,
    num_steps=30,
    guidance_scale=7.5,
    traverse_subdirs=False
)
```


## Parameters

* model_id: Identifier of the diffusion model (e.g., "stabilityai/stable-diffusion-2-1-base").
* scheduler_type: Scheduler type ('DPMS', 'DDIM', 'DDIMInverse').
* key_hex: 64-character hexadecimal string for encryption.
* nonce_hex: 32-character hexadecimal string for encryption.
* use_seed: Boolean to use a fixed seed for reproducibility.
* random_seed: Integer seed value if use_seed is True.
* seed_mode: Seed handling in batch processing ('sequential' or 'random').
* device: Computation device ('cuda' or 'cpu').
* turnoffWatermark: If True, disables watermark embedding.
* watermarkoncpu: If True, performs watermark operations on CPU.
* dtype: Tensor data type (e.g., torch.float32).
* message: Watermark message to embed.
* message_length: Bit length of the watermark message.
* window_size: Window size for watermark embedding and extraction.
* threshold: Accuracy threshold for watermark extraction in batch processing.


## Examples

Refer to the `example_usage` function in the `main.py` for a complete demonstration:

1. Initialize the engine.
2. Generate an image with a watermark.
3. Extract the watermark from the generated image.
4. Process a directory of images to extract and evaluate watermarks.

## Note

To generate images without watermarks, set `turnoffWatermark=True` during engine initialization.
