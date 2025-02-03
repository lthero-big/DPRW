# Diversity-Preserving Robust Watermarking (DPRW)

This repository provides the reference implementation of the paper *"Diversity-Preserving Robust Watermarking for Diffusion Model Generated Images"*. 

Our project aims to embed watermarks into the latent noise space of diffusion models, striking a balance between image fidelity and diversity while maintaining high robustness against common image distortions (e.g., Gaussian noise, JPEG compression, resizing, cropping).

## Features

- **High-Capacity Bit-Level Watermark Embedding**: Directly operates on binary bits in the latent noise, allowing flexible watermark capacity settings.  
- **No Model Modification**: Does not require fine-tuning or altering the pretrained diffusion model; watermark insertion only occurs in the noise phase.  
- **Preservation of Statistical Properties**: By binarizing the noise and using a random offset mapping, the Gaussian distribution properties are largely maintained, minimizing the impact on image quality and diversity.  
- **Robustness Verification**: Demonstrates high extraction accuracy under various distortions (compression, blurring, noise, cropping, etc.).  
