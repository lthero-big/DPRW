import argparse
import os
import glob
from typing import Tuple, Optional, List
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from datetime import datetime
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, DDIMScheduler,DDIMInverseScheduler, AutoencoderKL
from torchvision import transforms as tvt
from scipy.stats import norm, kstest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
import logging


# 日志工具类
class Loggers:
    """日志管理类，单例模式"""
    _logger = None
    
    @classmethod
    def get_logger(cls, log_dir: str = './logs') -> 'logging.Logger':
        if cls._logger is None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M")}.log')
            cls._logger = logging.getLogger('DPRW_Engine')
            cls._logger.setLevel(logging.INFO)
            cls._logger.handlers.clear()
            formatter = logging.Formatter("%(asctime)s %(levelname)s: [%(name)s] %(message)s")
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            console_handler = logging.StreamHandler()
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)
            cls._logger.addHandler(console_handler)
        return cls._logger

class WatermarkUtils:
    @staticmethod
    def calculate_bit_accuracy(original_hex: str, extracted_bin: str) -> Tuple[str, float]:
        orig_bin = bin(int(original_hex, 16))[2:].zfill(len(original_hex) * 4)
        min_len = min(len(orig_bin), len(extracted_bin))
        accuracy = sum(a == b for a, b in zip(orig_bin[:min_len], extracted_bin[:min_len])) / min_len
        return orig_bin, accuracy

# DPRW 主引擎类
class DPRW_Engine:
    """DPRW 水印引擎，整合嵌入、提取、评估、图像生成和逆向功能"""
    def __init__(self, model_id: str = 'stabilityai/stable-diffusion-2-1-base', scheduler_type: str = 'DDIM',
                    key_hex: Optional[str] = None, nonce_hex: Optional[str] = None,
                    use_seed: bool = False, random_seed: int = 42, device: str = 'cuda',turnoffWatermark=False,
                    watermarkoncpu:bool=False,dtype: torch.dtype = torch.float16, solver_order: int = 2, inv_order: int = 0,
                    seed_mode: str = 'sequential', log_dir: str = './logs'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.watermarkoncpu=watermarkoncpu
        self.dtype = dtype
        self.logger = Loggers.get_logger(log_dir)
        self.model_id = model_id
        self.scheduler_type = scheduler_type.upper()
        self.solver_order = solver_order
        self.inv_order = inv_order
        self.use_seed = use_seed
        self.random_seed = random_seed
        self.seed_mode = seed_mode  # 新增参数：'sequential' 或 'random'
        self.rng = np.random.RandomState(random_seed) if use_seed else np.random
        self.key, self.nonce = self._init_crypto(key_hex, nonce_hex)
        self.turnoffWatermark:bool = turnoffWatermark
        self.logger.info(f"Initialized - DPRW Engine with model: {model_id}")
        self.logger.info(f"Initialized - Scheduler: {scheduler_type}")
        self.logger.info(f"Initialized - Key: {self.key.hex()}")
        self.logger.info(f"Initialized - Nonce: {self.nonce.hex()}")

    # 初始化方法
    def _init_crypto(self, key_hex: Optional[str], nonce_hex: Optional[str]) -> Tuple[bytes, bytes]:
        """初始化密钥和 nonce"""
        key = self._validate_hex(key_hex, 64, 'key_hex', os.urandom(32))
        nonce = self._validate_hex(nonce_hex, 32, 'nonce_hex', os.urandom(16))
        return key, nonce

    def _validate_hex(self, hex_str: Optional[str], length: int, name: str, default: bytes) -> bytes:
        """验证并生成十六进制字符串"""
        if hex_str and len(hex_str) == length and all(c in '0123456789abcdefABCDEF' for c in hex_str):
            return bytes.fromhex(hex_str)
        self.logger.warning(f"Invalid {name}: {hex_str}, generating new random value for {name}")
        return default
    
    def _create_pipeline(self, scheduler_type: str) -> StableDiffusionPipeline:
        """根据调度器类型动态创建管道"""
        if scheduler_type == "DPMS":
            scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder='scheduler')
        elif scheduler_type == "DDIM":
            scheduler = DDIMScheduler.from_pretrained(self.model_id, subfolder='scheduler')
        elif scheduler_type == "DDIMInverse":
            scheduler = DDIMInverseScheduler.from_pretrained(self.model_id, subfolder='scheduler')
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,  # 关键参数
            device_map=None           # 关键参数
        )
        return pipe.to(self.device)
    
    @staticmethod
    def _disabled_safety_checker(images, clip_input):
        return images, [False] * images.shape[0] if len(images.shape) == 4 else False

    @staticmethod
    def _load_image(img_path: str, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        pil_img = Image.open(img_path).convert('RGB')
        if target_size:
            pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        return tvt.ToTensor()(pil_img)[None, ...]

    @staticmethod
    def _img_to_latents(x: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
        x = 2. * x - 1.
        posterior = vae.encode(x).latent_dist
        return posterior.mean * 0.18215

    # 水印嵌入方法
    def _create_watermark(self, total_blocks: int, message: str, message_length: int) -> bytes:
        """生成水印字节"""
        length_bits = message_length if message_length > 0 else 128
        length_bytes = length_bits // 8
        msg_bytes = message.encode('utf-8')
        padded_msg = msg_bytes.ljust(length_bytes, b'\x00')[:length_bytes]
        repeats = total_blocks // length_bits
        watermark = padded_msg * repeats + b'\x00' * ((total_blocks % length_bits) // 8)
        self.logger.info(f"Create watermark - Message: {message}")
        self.logger.info(f"Create watermark - Message Length: {message_length}")
        self.logger.info(f"Create watermark - Watermark repeats: {repeats} times")
        return watermark

    def _encrypt(self, watermark: bytes) -> str:
        """加密水印"""
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(watermark) + encryptor.finalize()
        return ''.join(format(byte, '08b') for byte in encrypted)

    def _binarize_noise(self, noise: torch.Tensor) -> np.ndarray:
        """将噪声二值化"""
        # 取出 batch 维度，计算 CDF
        noise_np = noise[0].cpu().numpy()
        return (norm.cdf(noise_np) > 0.5).astype(np.uint8).flatten()

    # old but works fine, which works on cpu
    def _embed_bits_cpu(self, binary: np.ndarray, bits: str, window_size: int) -> np.ndarray:
        """嵌入水印位"""
        if isinstance(binary, torch.Tensor):
            binary = binary.cpu().numpy()
        binary = binary.copy()
        for i in range(0, len(binary), window_size):
            window_end = min(i + window_size, len(binary))
            window_sum = np.sum(binary[i:window_end])
            bit_idx = i // window_size
            if bit_idx < len(bits):
                target_parity = int(bits[bit_idx])
                if window_sum % 2 != target_parity:
                    mid_idx = i + (window_end - i) // 2
                    if mid_idx < len(binary):
                        binary[mid_idx] = 1 - binary[mid_idx]
        return binary

    def _embed_bits_gpu(self, binary: torch.Tensor, bits: str, window_size: int) -> torch.Tensor:
        """嵌入水印位（完全向量化，GPU 上）"""
        binary = torch.from_numpy(binary).clone().to(self.device)
        num_windows = len(binary) // window_size
        bits_tensor = torch.tensor([int(b) for b in bits[:num_windows]], dtype=torch.uint8, device=self.device)
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        window_sums = windows.sum(dim=1) % 2  # 计算每个窗口的奇偶性
        flip_mask = window_sums != bits_tensor  # 需要翻转的窗口
        mid_idx = window_size // 2
        flip_indices = (torch.arange(num_windows, device=self.device) * window_size + mid_idx)[flip_mask]
        binary[flip_indices] = 1 - binary[flip_indices]
        return binary
    
    # works fine on both cpu and gpu
    def _embed_bits(self, binary: torch.Tensor, bits: str, window_size: int) -> torch.Tensor:
        if self.device.type == 'cpu' or self.watermarkoncpu:
            self.logger.info("Embedding bits on CPU")
            return self._embed_bits_cpu(binary, bits, window_size)
        elif self.device.type == 'cuda':
            self.logger.info("Embedding bits on GPU")
            return self._embed_bits_gpu(binary, bits, window_size)

    def _Gaussian_test(self, noise: torch.Tensor) -> None:
        if isinstance(noise, torch.Tensor):
            noise = noise.cpu().numpy()
        samples = noise.flatten()
        _, p_value = kstest(samples, 'norm', args=(0, 1))
        if np.isnan(samples).any() or np.isinf(samples).any():
            raise ValueError("Restored noise contains NaN or Inf values")
        if np.var(samples) == 0:
            raise ValueError("Restored noise variance is 0")
        if p_value < 0.05:
            raise ValueError(f"Restored noise failed Gaussian test (p={p_value:.4f})")
        self.logger.info(f"Gaussian test passed: p={p_value:.4f}")

    # old method which works on cpu and works fine
    def _restore_noise_cpu(self, binary: np.ndarray, shape: Tuple[int, ...], check_gaussian: bool = True) -> torch.Tensor:
        """还原高斯噪声，保留未修改的原始值"""
        if isinstance(binary, torch.Tensor):
            binary = binary.cpu().numpy()
        noise_np = self.init_noise.clone().cpu().numpy()[0]  # 复制原始噪声，移除 batch 维度
        binary_reshaped = binary.reshape(4, shape[2], shape[3])
        original_binary = (norm.cdf(noise_np) > 0.5).astype(np.uint8)

        for c in range(4):
            for h in range(shape[2]):
                for w in range(shape[3]):
                    binary_value = binary_reshaped[c, h, w]
                    original_binary_value = original_binary[c, h, w]
                    if binary_value != original_binary_value:
                        u = self.rng.uniform(0, 0.5 - 1e-8) if self.use_seed else np.random.uniform(0, 0.5 - 1e-8)
                        theta = u + binary_value * 0.5
                        noise_np[c, h, w] = norm.ppf(theta)
        res = torch.tensor(noise_np, dtype=self.dtype, device=self.device).unsqueeze(0)
        # 高斯检验
        if check_gaussian:
            self._Gaussian_test(res)
        return res
        
    #     return noise
    def _restore_noise_gpu(self, binary: torch.Tensor, shape: Tuple[int, ...], check_gaussian: bool = True) -> torch.Tensor:
        noise = self.init_noise.clone().to(self.device)
        binary_reshaped = binary.view(4, shape[2], shape[3])
        noise_fp32 = noise.to(torch.float32)
        original_binary = (torch.sigmoid(noise_fp32) > 0.5).to(torch.uint8)
        mask = binary_reshaped != original_binary
        
        # 设置随机种子并生成随机数 u
        # if self.use_seed:
        #     torch.manual_seed(self.random_seed)
        u = torch.rand_like(noise_fp32, device=self.device) * 0.5
        
        theta = u + binary_reshaped.float() * 0.5
        adjustment = torch.erfinv(2 * theta[mask] - 1) * torch.sqrt(torch.tensor(2.0, device=self.device, dtype=torch.float32))
        noise_fp32[mask] = adjustment
        noise = noise_fp32.to(self.dtype)
        
        if check_gaussian:
            self._Gaussian_test(noise)
        
        return noise
    
    # works fine on both cpu and gpu
    def _restore_noise(self, binary: torch.Tensor, shape: Tuple[int, ...], check_gaussian: bool = True) -> torch.Tensor:
        if self.device.type == 'cpu' or self.watermarkoncpu:
            self.logger.info("Restoring noise on CPU")
            return self._restore_noise_cpu(binary, shape)
        elif self.device.type == 'cuda':
            self.logger.info("Restoring noise on GPU")
            return self._restore_noise_gpu(binary, shape, check_gaussian)

    def get_embed_watermark(self, noise: torch.Tensor, message: str, message_length: int = 256, window_size: int = 1,batchsize:int=1) -> Tuple[torch.Tensor, List[bytes]]:
        """嵌入水印"""
        total_blocks = noise.numel() // (noise.shape[0] * window_size)
        watermark = self._create_watermark(total_blocks, message, message_length)
        encrypted_bits = self._encrypt(watermark)
        binary = self._binarize_noise(noise)
        binary_embedded = self._embed_bits(binary, encrypted_bits, window_size)
        restored_noise = self._restore_noise(binary_embedded, noise.shape)
        
        return restored_noise

    # 水印提取方法
    def _decrypt(self, encrypted: bytes) -> bytes:
        """解密水印"""
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted) + decryptor.finalize()

    # cpu version
    def extract_watermark_cpu(self, noise: torch.Tensor, message_length: int, window_size: int) -> Tuple[str, str]:
        """提取水印"""
        binary = (norm.cdf(noise.cpu().numpy()) > 0.5).astype(np.uint8).flatten()
        bits = [str(int(np.sum(binary[i:i + window_size]) % 2)) 
                for i in range(0, len(binary), window_size)]
        bit_str = ''.join(bits)
        byte_data = bytes(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str) - 7, 8))
        decrypted = self._decrypt(byte_data)
        all_bits = ''.join(format(byte, '08b') for byte in decrypted)
        segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits) - message_length + 1, message_length)]
        msg_bin = ''.join('1' if sum(s[i] == '1' for s in segments) > len(segments) / 2 else '0' 
                          for i in range(message_length))
        msg = bytes(int(msg_bin[i:i + 8], 2) for i in range(0, len(msg_bin), 8)).decode('utf-8', errors='replace')
        return msg_bin, msg
    
    # gpu version
    def extract_watermark_gpu(self, noise: torch.Tensor, message_length: int, window_size: int) -> Tuple[str, str]:
        """提取水印（在 GPU 上执行）"""
        noise = noise.to(self.device)  # 确保噪声在 GPU 上
        binary = (torch.sigmoid(noise) > 0.5).to(torch.uint8).flatten()  # 在 GPU 上二值化

        # 在 GPU 上计算窗口奇偶性
        num_windows = len(binary) // window_size
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        window_sums = windows.sum(dim=1) % 2  # 计算每个窗口的奇偶性
        bits = window_sums.cpu().numpy().astype(str).tolist()  # 转换为字符串列表

        # 后续处理（解密和消息提取）
        bit_str = ''.join(bits)
        byte_data = bytes(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str) - 7, 8))
        decrypted = self._decrypt(byte_data)
        all_bits = ''.join(format(byte, '08b') for byte in decrypted)
        segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits) - message_length + 1, message_length)]
        msg_bin = ''.join('1' if sum(s[i] == '1' for s in segments) > len(segments) / 2 else '0' 
                        for i in range(message_length))
        msg = bytes(int(msg_bin[i:i + 8], 2) for i in range(0, len(msg_bin), 8)).decode('utf-8', errors='replace')
        
        return msg_bin, msg
    
    # works fine on both cpu and gpu
    def extract_watermark(self, noise: torch.Tensor, message_length: int, window_size: int) -> Tuple[str, str]:
        if self.device.type == 'cpu':
            self.logger.info("Extracting watermark on CPU")
            return self.extract_watermark_cpu(noise, message_length, window_size)
        elif self.device.type == 'cuda':
            self.logger.info("Extracting watermark on GPU")
            return self.extract_watermark_gpu(noise, message_length, window_size)

    # 水印准确性评估
    def evaluate_accuracy(self, original_msg: str, extracted_bin: str, msg_str:str="") -> float:
        """计算位准确率"""
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        min_len = min(len(orig_bin), len(extracted_bin))
        orig_bin, extracted_bin = orig_bin[:min_len], extracted_bin[:min_len]
        accuracy = sum(a == b for a, b in zip(orig_bin, extracted_bin)) / min_len
        self.logger.info(f"Evaluation - Original binary: {orig_bin}")
        self.logger.info(f"Evaluation - Extracted binary: {extracted_bin}")
        if accuracy > 0.9:
            self.logger.info(f"Evaluation - Extracted message: {msg_str}")
        self.logger.info(f"Evaluation - Bit accuracy: {accuracy}")
        return accuracy

    def generate_image(self, prompt: str, width: int, height: int, num_steps: int, 
                       message: str, message_length: int = 256, window_size: int = 1, 
                       batchsize: int = 1, output_path: str = "generated_image.png") -> None:
        """生成带水印的图像"""
        # 创建带有 DPMS 调度器的管道
        pipe = self._create_pipeline("DPMS")
        latent_height, latent_width = height // 8, width // 8
        # 设置初始随机种子
        for i in range(batchsize):
            if self.use_seed and self.seed_mode == 'sequential':
                self.random_seed=self.random_seed + i
            elif self.use_seed and self.seed_mode == 'random':
                self.random_seed=self.random_seed + np.random.randint(0, 1000000)
            if self.use_seed:
                torch.manual_seed(self.random_seed)
            self.logger.info(f"Generation - Random seed: {self.random_seed}")
            # 初始化噪声
            self.init_noise = torch.randn(1, 4, latent_height, latent_width, device=self.device, dtype=self.dtype)
            # 嵌入水印
            if self.turnoffWatermark:
                watermarked_noise = self.init_noise
            else:
                watermarked_noise = self.get_embed_watermark(self.init_noise, message, message_length, window_size, batchsize)
            # 生成图像
            image = pipe(
                prompt=prompt,
                latents=watermarked_noise,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                output_type='pil'
            ).images[0]
            # 保存图像
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(f"{output_path.replace('.png', f'_{i}.png')}")
            self.logger.info(f"Generation - Generated image saved to {output_path}")

        # 释放管道资源
        del pipe
        torch.cuda.empty_cache()

        return 

    
    def invert_image(self, image_path: str, width: int, height: int, num_steps: int, guidance_scale: float = 1.0) -> torch.Tensor:
        """逆向生成潜在表示以提取水印"""
        # 创建带有 DDIMInverseScheduler 的管道
        pipe = self._create_pipeline("DDIMInverse")

        # 加载图像
        img = self._load_image(image_path, (width, height)).to(device=self.device, dtype=self.dtype)
        self.logger.info(f"Inverting image - {image_path} with shape {img.shape}")

        # 转换为潜在表示
        latents = self._img_to_latents(img, pipe.vae)

        # 逆向生成
        reversed_latents, _ = pipe(
            prompt="",
            negative_prompt="",
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            output_type='latent',
            return_dict=False,
            num_inference_steps=num_steps,
            latents=latents
        )

        # 释放管道资源
        del pipe
        torch.cuda.empty_cache()

        return reversed_latents.cpu()

    # 批量处理方法
    def process_single_image(self, image_path: str, message_length: int, window_size: int, threshold: float, 
                            original_msg: str, width: int, height: int, num_steps: int, guidance_scale: float) -> dict:
        """处理单张图像"""
        latents = self.invert_image(image_path, width, height, num_steps, guidance_scale)
        extracted_bin, extracted_msg = self.extract_watermark(latents, message_length, window_size)
        bit_accuracy = self.evaluate_accuracy(original_msg, extracted_bin,extracted_msg)
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        result = {
            "original_message_bin": orig_bin,
            "extracted_message_bin": extracted_bin,
            "bit_accuracy": bit_accuracy,
        }
        return result

    def process_directory(self, dir_path: str, message_length: int, window_size: int, threshold: float, 
                          original_msg:str, width: int, height: int, num_steps: int, guidance_scale: float,
                          traverse_subdirs: bool = False):
        """处理目录或递归处理子目录"""
        if traverse_subdirs:
            with open(os.path.join(dir_path, "result.txt"), "a") as root_file:
                self._write_batch_info(root_file,  num_steps)
            for root, dirs, _ in os.walk(dir_path):
                for subdir in dirs:
                    self._process_single_directory(os.path.join(root, subdir), message_length, window_size, threshold, 
                                                  original_msg, width, height, num_steps, guidance_scale)
            with open(os.path.join(dir_path, "result.txt"), "a") as root_file:
                root_file.write("=" * 40 + "Batch End" + "=" * 40 + "\n\n")
        else:
            self._process_single_directory(dir_path, message_length, window_size, threshold,original_msg,  
                                          width, height, num_steps, guidance_scale)

    def _process_single_directory(self, dir_path: str, message_length: int, window_size: int, threshold: float, 
                                  original_msg: str, width: int, height: int, num_steps: int, guidance_scale: float):
        """处理单个目录"""
        image_files = glob.glob(os.path.join(dir_path, "*.png")) + glob.glob(os.path.join(dir_path, "*.jpg"))
        if not image_files:
            return

        result_file_path = os.path.join(dir_path, "result.txt")
        with open(result_file_path, "a") as result_file:
            self._write_batch_info(result_file,  num_steps)
            total_bit_accuracy, recognized_count, processed_count = 0, 0, 0

            for image_path in tqdm(image_files):
                try:
                    result = self.process_single_image(image_path, message_length, window_size, threshold, original_msg,
                                                      width, height, num_steps, guidance_scale)
                    total_bit_accuracy += result["bit_accuracy"]
                    result_file.write(f"{os.path.basename(image_path)}, Bit Accuracy, {result['bit_accuracy']}\n")
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    result_file.write(f"Error processing {image_path}: {e}\n")

            if processed_count > 0:
                avg_bit_accuracy = total_bit_accuracy / processed_count
                identification_accuracy = recognized_count / processed_count

                self.logger.info(f"Batch processing completed. {image_path} Avg Bit Accuracy: {avg_bit_accuracy}")
                result_file.write(f"Average Bit Accuracy, {avg_bit_accuracy}\n")
                result_file.write("=" * 40 + "Batch End" + "=" * 40 + "\n")

                parent_dir = os.path.dirname(dir_path)
                with open(os.path.join(parent_dir, "result.txt"), "a") as parent_file:
                    parent_file.write(f"{os.path.basename(dir_path)}, Avg Bit Accuracy, {avg_bit_accuracy}\n")

    def _write_batch_info(self, file,  num_steps: int):
        """写入批次信息"""
        file.write("=" * 40 + "Batch Info" + "=" * 40 + "\n")
        file.write(f"Time, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Key Hex, {self.key.hex()}\n")
        file.write(f"Nonce Hex, {self.nonce.hex()}\n")
        file.write(f"Num Inference Steps, {num_steps}\n")
        file.write(f"Scheduler, {self.scheduler_type}\n")
        file.write("=" * 40 + "Batch Start" + "=" * 40 + "\n")

