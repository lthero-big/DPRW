# DPRW_Engine - Watermarking and Image Processing Engine

`DPRW_Engine` 是一个集成的 Python 类，用于在图像生成过程中嵌入和提取隐性水印，并支持从图像逆向提取潜在表示和批量处理图像目录。基于 Stable Diffusion 模型，支持多种调度器（DDIM、DPMS、DPMS-INV）。

## 功能
1. **水印嵌入**：将消息嵌入到噪声中。
2. **水印提取**：从噪声或图像提取水印。
3. **准确性评估**：计算提取水印的位准确率。
4. **图像生成**：生成带水印的图像并保存。
5. **图像逆向**：从图像逆向生成潜在表示。
6. **批量处理**：处理图像目录并生成统计报告。

## 依赖
- Python 3.8+
- torch
- diffusers
- numpy
- PIL (Pillow)
- tqdm
- scipy
- cryptography
- src.stable_diffusion.inverse_stable_diffusion (需手动安装)

安装依赖：
```bash
pip install torch diffusers numpy pillow tqdm scipy cryptography
```

## 初始化
```
from dprw_engine import DPRW_Engine

engine = DPRW_Engine(
    model_id="stabilityai/stable-diffusion-2-1-base",
    scheduler_type="DDIM",
    key_hex="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7",
    nonce_hex="05072fd1c2265f6f2e2a4080a2bfbdd8",
    use_seed=True,
    random_seed=42,
    device="cuda"  # 或 "cpu"
)
```
### 参数
* model_id: Stable Diffusion 模型 ID。
* scheduler_type: 调度器类型 ("DDIM", "DPMS", "DPMS-INV")。
* key_hex, nonce_hex: 加密密钥和 nonce（可选，默认为随机生成）。
* use_seed, random_seed: 是否使用种子和随机种子值。
* device: 运行设备 ("cuda" 或 "cpu")。
* dtype: 数据类型 (默认 torch.float16)。
* solver_order, inv_order: 调度器和逆向参数。

## 使用方法
### 1. 生成带水印的图像
```
prompt = "A girl in a white sweater at Times Square"
image_path, params = engine.generate_image(
    prompt=prompt, width=1024, height=1024, num_steps=30,
    message="lthero", message_length=256, window_size=1,
    output_path="output/generated_image.png"
)
print(f"Generated image saved to {image_path}")
```

### 2. 从图像逆向提取水印
```
latents = engine.invert_image("output/generated_image.png", width=1024, height=1024, num_steps=30, guidance_scale=7.5)
extracted_bin, extracted_msg = engine.extract_watermark(latents, message_length=256, window_size=1)
accuracy = engine.evaluate_accuracy("lthero", extracted_bin)
print(f"Extracted message: {extracted_msg}, Bit accuracy: {accuracy}")
```

### 3. 嵌入水印到噪声并提取
``` 
noise = torch.randn(1, 4, 128, 128, device=engine.device, dtype=engine.dtype)
watermarked_noise, _ = engine.embed_watermark(noise, "lthero", message_length=256, window_size=1)
extracted_bin, extracted_msg = engine.extract_watermark(watermarked_noise, message_length=256, window_size=1)
accuracy = engine.evaluate_accuracy("lthero", extracted_bin)
print(f"Extracted message: {extracted_msg}, Bit accuracy: {accuracy}")
```

### 4. 批量处理图像目录
```
engine.process_directory(
    "images", message_length=256, window_size=1, threshold=0.7,
    original_msg_hex="6c746865726f00000000", width=1024, height=1024, num_steps=30, guidance_scale=7.5,
    traverse_subdirs=False
)
print("Batch processing completed. Check results in images/result.txt")
```

### 5. 完整工作流：生成 -> 逆向 -> 提取
```
# 生成图像
image_path, _ = engine.generate_image(
    prompt="A futuristic cityscape", width=512, height=512, num_steps=50,
    message="secret", message_length=128, window_size=1, output_path="output/city.png"
)

# 逆向提取
latents = engine.invert_image(image_path, width=512, height=512, num_steps=50, guidance_scale=7.5)
extracted_bin, extracted_msg = engine.extract_watermark(latents, message_length=128, window_size=1)
accuracy = engine.evaluate_accuracy("secret", extracted_bin)
print(f"Extracted message: {extracted_msg}, Bit accuracy: {accuracy}")
```

## 输出
* 生成图像：保存为 PNG 文件，返回路径和加密参数。
* 提取：返回二进制字符串和解码消息。
* 批量处理：生成 result.txt，包含每张图像的位准确率和统计信息。

## 注意事项
* 确保图像尺寸与模型兼容（通常为 512x512 或 1024x1024）。
* 对于 DPMS-INV 调度器，需确保 src.stable_diffusion.inverse_stable_diffusion 可用。
* 日志文件保存在 log_dir 指定的目录中。

##扩展
* 修改 _init_pipeline 以支持其他调度器。
* 在 generate_image 中添加更多生成参数（如负提示词）。



