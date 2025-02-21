import numpy as np
import torch
import os
import logging
from typing import Tuple, Optional, List
from scipy.stats import norm, kstest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from datetime import datetime

class Loggers:
    _logger = None

    @classmethod
    def get_logger(cls, log_dir: str = './logs') -> logging.Logger:
        if cls._logger is None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M")}.log')
            cls._logger = logging.getLogger('WatermarkLogger')
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

class WatermarkBase:
    def __init__(self, device: str = 'cpu', log_dir: str = './logs'):
        self.device = torch.device(device)
        self.logger = Loggers.get_logger(log_dir)

    def _log(self, level: str, module: str, key: str, value: str):
        getattr(self.logger, level)(f"[{module}] {key}: {value}")

    def log_info(self, module: str, key: str, value: str):
        self._log('info', module, key, value)

    def log_warning(self, module: str, key: str, value: str):
        self._log('warning', module, key, value)

    def log_error(self, module: str, key: str, value: str):
        self._log('error', module, key, value)

class DPRW_WatermarkEmbed(WatermarkBase):
    def __init__(self, key_hex: Optional[str] = None, nonce_hex: Optional[str] = None,
                 use_seed: bool = False, random_seed: int = 42,
                 device: str = 'cpu', log_dir: str = './logs'):
        super().__init__(device, log_dir)
        self.use_seed = use_seed
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed) if use_seed else np.random
        self.key, self.nonce = self._init_crypto(key_hex, nonce_hex)

    def _init_crypto(self, key_hex: Optional[str], nonce_hex: Optional[str]) -> Tuple[bytes, bytes]:
        key = self._validate_hex(key_hex, 64, 'key_hex', os.urandom(32))
        nonce = self._validate_hex(nonce_hex, 32, 'nonce_hex', os.urandom(16))
        return key, nonce

    def _validate_hex(self, hex_str: Optional[str], length: int, name: str, default: bytes) -> bytes:
        if hex_str and len(hex_str) == length and all(c in '0123456789abcdefABCDEF' for c in hex_str):
            return bytes.fromhex(hex_str)
        self.log_warning("WatermarkEmbed", f"Invalid {name}", f"{hex_str}, generating random value")
        return default

    def _create_watermark(self, total_blocks: int, message: str, message_length: int) -> bytes:
        length_bits = message_length if message_length > 0 else 128
        length_bytes = length_bits // 8
        msg_bytes = message.encode('utf-8')
        padded_msg = msg_bytes.ljust(length_bytes, b'\x00')[:length_bytes]
        repeats = total_blocks // length_bits
        watermark = padded_msg * repeats + b'\x00' * ((total_blocks % length_bits) // 8)
        self.log_info("WatermarkEmbed", "Watermark repeats", str(repeats))
        return watermark

    def _encrypt(self, watermark: bytes) -> str:
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(watermark) + encryptor.finalize()
        return ''.join(format(byte, '08b') for byte in encrypted)

    def _binarize_noise(self, noise: torch.Tensor) -> np.ndarray:
        noise_np = noise.cpu().numpy()[-1] 
        return (norm.cdf(noise_np) > 0.5).astype(np.uint8).flatten()

    def _embed_bits(self, binary: np.ndarray, bits: str, window_size: int) -> np.ndarray:
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

    def _restore_noise(self, binary: np.ndarray, shape: Tuple[int, ...]) -> torch.Tensor:
        noise_np = np.zeros(shape[1:], dtype=np.float32) 
        binary_reshaped = binary.reshape(shape[1:])
        for c in range(shape[1]):
            for h in range(shape[2]):
                for w in range(shape[3]):
                    u = self.rng.uniform(0, 0.5 - 1e-8)
                    theta = u + binary_reshaped[c, h, w] * 0.5
                    noise_np[c, h, w] = norm.ppf(theta)

        samples = noise_np.flatten()
        _, p_value = kstest(samples, 'norm', args=(0, 1))
        if p_value < 0.05:
            raise ValueError(f"Restored noise failed Gaussian test (p={p_value:.4f})")
        self.log_info("WatermarkEmbed", "Gaussian test passed", f"p={p_value:.4f}")
        return torch.tensor(noise_np, dtype=torch.float32, device=self.device)

    def embed(self, noise: torch.Tensor, message: str, message_length: int = -1, window_size: int = 1) -> Tuple[torch.Tensor, List[bytes]]:
        total_blocks = noise.numel() // (noise.shape[0] * window_size)
        watermark = self._create_watermark(total_blocks, message, message_length)
        encrypted_bits = self._encrypt(watermark)
        binary = self._binarize_noise(noise)
        binary_embedded = self._embed_bits(binary, encrypted_bits, window_size)
        restored_noise = self._restore_noise(binary_embedded, noise.shape)
        self.log_info("WatermarkEmbed", "Key", self.key.hex())
        self.log_info("WatermarkEmbed", "Nonce", self.nonce.hex())
        self.log_info("WatermarkEmbed", "Message", message)
        return restored_noise, [self.key, self.nonce]


class DPRW_WatermarkExtract(WatermarkBase):
    def __init__(self, key_hex: str, nonce_hex: str, device: str = 'cpu', log_dir: str = './logs'):
        super().__init__(device, log_dir)
        self.key = bytes.fromhex(key_hex)
        self.nonce = bytes.fromhex(nonce_hex)

    def _decrypt(self, encrypted: bytes) -> bytes:
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted) + decryptor.finalize()

    def extract(self, noise: torch.Tensor, message_length: int, window_size: int) -> Tuple[str, str]:
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
        self.log_info("WatermarkExtract", "Extracted binary", msg_bin)
        self.log_info("WatermarkExtract", "Extracted message", msg)
        return msg_bin, msg

class DPRW_WatermarkAccuracy(WatermarkBase):
    def evaluate(self, original_msg: str, extracted_bin: str) -> float:
        orig_bin = bin(int(original_msg.encode('utf-8').hex(), 16))[2:].zfill(len(original_msg) * 8)
        min_len = min(len(orig_bin), len(extracted_bin))
        orig_bin, extracted_bin = orig_bin[:min_len], extracted_bin[:min_len]
        accuracy = sum(a == b for a, b in zip(orig_bin, extracted_bin)) / min_len
        self.log_info("WatermarkAccuracy", "Original binary", orig_bin)
        self.log_info("WatermarkAccuracy", "Extracted binary", extracted_bin)
        self.log_info("WatermarkAccuracy", "Bit accuracy", f"{accuracy:.4f}")
        return accuracy

