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


class Loggers:
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

class DPRW_Engine:
    def __init__(self, model_id: str = 'stabilityai/stable-diffusion-2-1-base', scheduler_type: str = 'DDIM',
                 key_hex: Optional[str] = None, nonce_hex: Optional[str] = None,
                 use_seed: bool = False, random_seed: int = 42, device: str = 'cuda',
                 dtype: torch.dtype = torch.float32, solver_order: int = 2, inv_order: int = 0,
                 log_dir: str = './logs'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.logger = Loggers.get_logger(log_dir)
        self.model_id = model_id
        self.scheduler_type = scheduler_type.upper()
        self.solver_order = solver_order
        self.inv_order = inv_order
        self.use_seed = use_seed
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed) if use_seed else np.random
        self.key, self.nonce = self._init_crypto(key_hex, nonce_hex)
        self.logger.info(f"Initialized - DPRW Engine with model: {model_id}")
        self.logger.info(f"Initialized - Scheduler: {scheduler_type}")
        self.logger.info(f"Initialized - Key: {self.key.hex()}")
        self.logger.info(f"Initialized - Nonce: {self.nonce.hex()}")

    def _init_crypto(self, key_hex: Optional[str], nonce_hex: Optional[str]) -> Tuple[bytes, bytes]:
        key = self._validate_hex(key_hex, 64, 'key_hex', os.urandom(32))
        nonce = self._validate_hex(nonce_hex, 32, 'nonce_hex', os.urandom(16))
        return key, nonce

    def _validate_hex(self, hex_str: Optional[str], length: int, name: str, default: bytes) -> bytes:
        if hex_str and len(hex_str) == length and all(c in '0123456789abcdefABCDEF' for c in hex_str):
            return bytes.fromhex(hex_str)
        self.logger.warning(f"Invalid {name}: {hex_str}, generating new random value for {name}")
        return default
    
    def _create_pipeline(self, scheduler_type: str) -> StableDiffusionPipeline:
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
            low_cpu_mem_usage=False,  
            device_map=None           
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

    def _create_watermark(self, total_blocks: int, message: str, message_length: int) -> bytes:
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
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(watermark) + encryptor.finalize()
        return ''.join(format(byte, '08b') for byte in encrypted)

    def _binarize_noise(self, noise: torch.Tensor) -> np.ndarray:
        noise_np = noise[0].cpu().numpy()
        return (norm.cdf(noise_np) > 0.5).astype(np.uint8).flatten()

    def _embed_bits(self, binary: torch.Tensor, bits: str, window_size: int) -> torch.Tensor:
        binary = torch.from_numpy(binary).clone().to(self.device)
        num_windows = len(binary) // window_size
        bits_tensor = torch.tensor([int(b) for b in bits[:num_windows]], dtype=torch.uint8, device=self.device)
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        window_sums = windows.sum(dim=1) % 2  
        flip_mask = window_sums != bits_tensor  
        mid_idx = window_size // 2
        flip_indices = (torch.arange(num_windows, device=self.device) * window_size + mid_idx)[flip_mask]
        binary[flip_indices] = 1 - binary[flip_indices]
        return binary

    def _restore_noise(self, binary: torch.Tensor, shape: Tuple[int, ...],fix_batchsize:int =1, check_gaussian: bool = False) -> torch.Tensor:
        noise = self.init_noise.clone().to(self.device)
        binary_reshaped = binary.view(4, shape[2], shape[3])
        original_binary = (torch.sigmoid(noise) > 0.5).to(torch.uint8)
        mask = binary_reshaped != original_binary
        u = torch.rand_like(noise, device=self.device) * 0.5
        theta = u + binary_reshaped.float() * 0.5
        noise[mask] = torch.erfinv(2 * theta[mask] - 1).to(noise.dtype) * torch.sqrt(torch.tensor(2.0, device=self.device,dtype=noise.dtype))
        if check_gaussian:
            samples = noise.flatten().cpu().numpy()
            _, p_value = kstest(samples, 'norm', args=(0, 1))
            if p_value < 0.05:
                self.logger.warning(f"Restored noise failed Gaussian test (p={p_value:.4f})")
        return noise

    def embed_watermark(self, noise: torch.Tensor, message: str, message_length: int = 256, window_size: int = 1,fix_batchsize:int=1) -> Tuple[torch.Tensor, List[bytes]]:
        total_blocks = noise.numel() // (noise.shape[0] * window_size)
        watermark = self._create_watermark(total_blocks, message, message_length)
        encrypted_bits = self._encrypt(watermark)
        binary = self._binarize_noise(noise)
        binary_embedded = self._embed_bits(binary, encrypted_bits, window_size)
        restored_noise = self._restore_noise(binary_embedded, noise.shape,fix_batchsize)
        
        
        return restored_noise

    def _decrypt(self, encrypted: bytes) -> bytes:
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted) + decryptor.finalize()

    def extract_watermark(self, noise: torch.Tensor, message_length: int, window_size: int) -> Tuple[str, str]:
        noise = noise.to(self.device) 
        binary = (torch.sigmoid(noise) > 0.5).to(torch.uint8).flatten()


        num_windows = len(binary) // window_size
        windows = binary[:num_windows * window_size].view(num_windows, window_size)
        window_sums = windows.sum(dim=1) % 2 
        bits = window_sums.cpu().numpy().astype(str).tolist()

        bit_str = ''.join(bits)
        byte_data = bytes(int(bit_str[i:i + 8], 2) for i in range(0, len(bit_str) - 7, 8))
        decrypted = self._decrypt(byte_data)
        all_bits = ''.join(format(byte, '08b') for byte in decrypted)
        segments = [all_bits[i:i + message_length] for i in range(0, len(all_bits) - message_length + 1, message_length)]
        msg_bin = ''.join('1' if sum(s[i] == '1' for s in segments) > len(segments) / 2 else '0' 
                        for i in range(message_length))
        msg = bytes(int(msg_bin[i:i + 8], 2) for i in range(0, len(msg_bin), 8)).decode('utf-8', errors='replace')
        
        return msg_bin, msg
    
    def evaluate_accuracy(self, original_msg: str, extracted_bin: str,msg_str:str) -> float:
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
                       fix_batchsize: int = 1, output_path: str = "generated_image.png") -> None:
        pipe = self._create_pipeline("DPMS")

        latent_height, latent_width = height // 8, width // 8
        self.init_noise = torch.randn(1, 4, latent_height, latent_width, device=self.device, dtype=self.dtype)

        watermarked_noise = self.embed_watermark(self.init_noise, message, message_length, window_size, fix_batchsize)

        image = pipe(
            prompt=prompt,
            latents=watermarked_noise,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            output_type='pil'
        ).images[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        self.logger.info(f"Generation - Generated image saved to {output_path}")

        del pipe
        torch.cuda.empty_cache()

        return 
    
    def invert_image(self, image_path: str, width: int, height: int, num_steps: int, guidance_scale: float = 1.0) -> torch.Tensor:
        pipe = self._create_pipeline("DDIMInverse")

        img = self._load_image(image_path, (width, height)).to(device=self.device, dtype=self.dtype)
        self.logger.info(f"Inverting image - {image_path} with shape {img.shape}")

        latents = self._img_to_latents(img, pipe.vae)

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

        del pipe
        torch.cuda.empty_cache()

        return reversed_latents.cpu()

    def process_single_image(self, image_path: str, message_length: int, window_size: int, threshold: float, 
                            original_msg_hex: str, width: int, height: int, num_steps: int, guidance_scale: float) -> dict:
        latents = self.invert_image(image_path, width, height, num_steps, guidance_scale)
        extracted_bin, extracted_msg = self.extract_watermark(latents, message_length, window_size)
        orig_bin, bit_accuracy = WatermarkUtils.calculate_bit_accuracy(original_msg_hex, extracted_bin)
        is_recognized = bit_accuracy >= threshold

        result = {
            "original_message_bin": orig_bin,
            "extracted_message_bin": extracted_bin,
            "bit_accuracy": bit_accuracy,
            "is_recognized": is_recognized
        }
        return result

    def process_directory(self, dir_path: str, message_length: int, window_size: int, threshold: float, 
                          original_msg_hex: str, width: int, height: int, num_steps: int, guidance_scale: float,
                          traverse_subdirs: bool = False):
        if traverse_subdirs:
            with open(os.path.join(dir_path, "result.txt"), "a") as root_file:
                self._write_batch_info(root_file, original_msg_hex, num_steps)
            for root, dirs, _ in os.walk(dir_path):
                for subdir in dirs:
                    self._process_single_directory(os.path.join(root, subdir), message_length, window_size, threshold, 
                                                  original_msg_hex, width, height, num_steps, guidance_scale)
            with open(os.path.join(dir_path, "result.txt"), "a") as root_file:
                root_file.write("=" * 40 + "Batch End" + "=" * 40 + "\n\n")
        else:
            self._process_single_directory(dir_path, message_length, window_size, threshold, original_msg_hex, 
                                          width, height, num_steps, guidance_scale)

    def _process_single_directory(self, dir_path: str, message_length: int, window_size: int, threshold: float, 
                                  original_msg_hex: str, width: int, height: int, num_steps: int, guidance_scale: float):
        image_files = glob.glob(os.path.join(dir_path, "*.png")) + glob.glob(os.path.join(dir_path, "*.jpg"))
        if not image_files:
            return

        result_file_path = os.path.join(dir_path, "result.txt")
        with open(result_file_path, "a") as result_file:
            self._write_batch_info(result_file, original_msg_hex, num_steps)
            total_bit_accuracy, recognized_count, processed_count = 0, 0, 0

            for image_path in tqdm(image_files):
                try:
                    result = self.process_single_image(image_path, message_length, window_size, threshold, 
                                                      original_msg_hex, width, height, num_steps, guidance_scale)
                    recognized_count += result["is_recognized"]
                    total_bit_accuracy += result["bit_accuracy"]
                    result_file.write(f"{os.path.basename(image_path)}, Bit Accuracy, {result['bit_accuracy']}\n")
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    result_file.write(f"Error processing {image_path}: {e}\n")

            if processed_count > 0:
                avg_bit_accuracy = total_bit_accuracy / processed_count
                identification_accuracy = recognized_count / processed_count

                result_file.write(f"Average Bit Accuracy, {avg_bit_accuracy}\n")
                result_file.write(f"Identification Accuracy, {identification_accuracy}\n")
                result_file.write("=" * 40 + "Batch End" + "=" * 40 + "\n")

                parent_dir = os.path.dirname(dir_path)
                with open(os.path.join(parent_dir, "result.txt"), "a") as parent_file:
                    parent_file.write(f"{os.path.basename(dir_path)}, Avg Bit Accuracy, {avg_bit_accuracy}, Identification Accuracy, {identification_accuracy}\n")

    def _write_batch_info(self, file, original_msg_hex: str, num_steps: int):
        file.write("=" * 40 + "Batch Info" + "=" * 40 + "\n")
        file.write(f"Time, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Key Hex, {self.key.hex()}\n")
        file.write(f"Nonce Hex, {self.nonce.hex()}\n")
        file.write(f"Original Message Hex, {original_msg_hex}\n")
        file.write(f"Num Inference Steps, {num_steps}\n")
        file.write(f"Scheduler, {self.scheduler_type}\n")
        file.write("=" * 40 + "Batch Start" + "=" * 40 + "\n")

def example_usage():
    engine = DPRW_Engine(
        # digiplay/majicMIX_realistic_v7
        # stabilityai/stable-diffusion-2-1-base
        model_id="digiplay/majicMIX_realistic_v7",
        scheduler_type="DDIM",
        key_hex="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7",
        nonce_hex="05072fd1c2265f6f2e2a4080a2bfbdd8",
        use_seed=True,
        random_seed=42,
        device="cuda" 
    )
    fix_batchsize = 1
    message = "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a75822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"
    width, height = 1024, 1024
    # you can set message_length to 0 to use the default value of 128
    message_length = len(message)*8
    window_size = 1
    num_steps = 30
    generated_image_path = "output/generated_image.png"

    prompt = "A girl in a white sweater at Times Square"
    engine.generate_image(
        prompt=prompt, width=width, height=height, num_steps=num_steps,
        message=message, message_length=message_length, window_size=window_size,fix_batchsize=1,
        output_path=generated_image_path
    )

    reversed_latents = engine.invert_image(generated_image_path, width=width, 
                                           height=height, 
                                           num_steps=num_steps, 
                                           guidance_scale=7.5)
    extracted_bin, extracted_msg = engine.extract_watermark(reversed_latents, 
                                                            message_length=message_length, 
                                                            window_size=window_size)
    accuracy = engine.evaluate_accuracy(message, extracted_bin,extracted_msg)

if __name__ == "__main__":
    example_usage()
