import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend
from datetime import datetime
import os
from scipy.stats import norm
import torch
import json
import logging
import time


class Loggers:
    def __init__(self, logger_file_path: str):
        self.logger_file_path = logger_file_path

    def create_logger(self) -> logging.Logger:
        # make dir if not exists
        if not os.path.exists(self.logger_file_path):
            os.makedirs(self.logger_file_path)

        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        final_log_file = os.path.join(self.logger_file_path, log_name)

        logger = logging.getLogger()  
        if logger.hasHandlers():
            print("clear logger handlers")
            logger.handlers.clear()
        logger.setLevel(logging.INFO)  # DEBUG/ERROR etc.

        file_handler = logging.FileHandler(final_log_file, mode='a', encoding='utf-8')
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger 

class Common:
    _logger = None

    def __init__(self, device:str ='cpu', logger_file_path:str='./logs'):
        self.device = device
        if Common._logger is None:
            Common._logger = Loggers(logger_file_path).create_logger()

    @property
    def logger(self) -> logging.Logger:
        return Common._logger
    
    def LogInfo(self, ModuleName:str, MessageKey:str, MessageValue:str):
        self.logger.info(f'[{ModuleName}] {MessageKey}: {MessageValue}')

    def LogWarning(self, ModuleName:str, MessageKey:str, MessageValue:str):
        self.logger.warning(f'[{ModuleName}] {MessageKey}: {MessageValue}')

    def LogError(self, ModuleName:str, MessageKey:str, MessageValue:str):
        self.logger.error(f'[{ModuleName}] {MessageKey}: {MessageValue}')


class DPRW_WatermarkEmbed(Common):
    def __init__(self, key_hex:str=None, nonce_hex:str=None, use_seed:int=0, 
                random_seed:int=42, device:str ='cpu', logger_file_path:str='./logs'):
        super().__init__(device, logger_file_path)
        self.key_hex = key_hex
        self.nonce_hex = nonce_hex
        self.use_seed = use_seed
        self.random_seed = random_seed
        self.ParamsNeedReturn=[]
        self.__initRandomSeed()
        self.__initKeyAndNonce()
        
    def __is_valid_hex(self,string:str, expected_length:int):
            if not isinstance(string, str):
                return False
            if len(string) != expected_length:
                return False
            try:
                bytes.fromhex(string)
                return True
            except ValueError:
                return False

    def __initRandomSeed(self):
        if self.use_seed == 1:
            self.rng = np.random.RandomState(seed=self.random_seed)
    
    def __initKeyAndNonce(self):
        if not self.__is_valid_hex(self.key_hex, 64):
            self.LogWarning("WatermarkEmbed",f"Invalid key_hex",f"{self.key_hex}, generating random key")
            self.key=os.urandom(32)
            self.key_hex=self.key.hex()
            self.ParamsNeedReturn.append(self.key)
        else:
            self.key = bytes.fromhex(self.key_hex)
            
        if not self.__is_valid_hex(self.nonce_hex, 32):
            self.LogWarning("WatermarkEmbed",f"Invalid nonce_hex",f"{self.nonce_hex}, generating random nonce")
            self.nonce=os.urandom(16)
            self.nonce_hex=self.nonce.hex()
            self.ParamsNeedReturn.append(self.nonce)
        else:
            self.nonce = bytes.fromhex(self.nonce_hex)

    def __initMessage(self,total_blocks_needed:int, message_length:int=-1) -> bytes:
        if message_length != -1:
            watermark_length_bits = message_length
        else:
            watermark_length_bits = 128

        length_of_msg_bytes = watermark_length_bits // 8
        
        if self.message:
            message_bytes = str(self.message).encode()
            if len(message_bytes) < length_of_msg_bytes:
                padded_message = message_bytes + b'\x00' * (length_of_msg_bytes - len(message_bytes))
            else:
                padded_message = message_bytes[:length_of_msg_bytes]
        else:
            padded_message = os.urandom(length_of_msg_bytes)

        repeats = total_blocks_needed // watermark_length_bits
        self.LogInfo("WatermarkEmbed",f"repeats",repeats)

        watermark_bytes = padded_message * repeats
        remaining_bits = total_blocks_needed * 8 - repeats * watermark_length_bits
        if remaining_bits > 0:
            watermark_bytes += b'\x00' * (remaining_bits // 8)

        return watermark_bytes
    

    def __EncryptMessage(self, watermark_bytes:bytes) -> str:
        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        m = encryptor.update(watermark_bytes) + encryptor.finalize()
        m_bits = ''.join(format(byte, '08b') for byte in m)
        return m_bits


    def __NoiseToBinary(self) -> np.ndarray:
        self.initialNoise_cpu = self.initialNoise.cpu().numpy()
        binary_matrix = (norm.cdf(self.initialNoise_cpu) > 0.5).astype(int)
        binary_matrix = binary_matrix[self.fix_batchsize - 1]

        binary_flat = binary_matrix.flatten()
        return binary_flat


    def __CoreMethod(self, binary_flat:np.ndarray, m_bits:str, total_blocks_needed:int) -> np.ndarray:
        bit_idx = 0
        total_m_bits = len(m_bits)
        count_fix_nums = 0
        total_elements = len(binary_flat)

        # self.LogInfo("total_elements",total_elements)
        # self.LogInfo("total_m_bits",total_m_bits)

        for i in range(0, total_elements, self.window_size):
            window_end = min(i + self.window_size, total_elements)  
            window_sum = np.sum(binary_flat[i:window_end])

            target_sum = window_sum
            if bit_idx < total_m_bits:
                if int(m_bits[bit_idx]) == 0:
                    if window_sum % 2 != 0:
                        target_sum -= 1
                else:
                    if window_sum % 2 == 0:
                        target_sum += 1

            if target_sum != window_sum:
                count_fix_nums += 1

                mid_index = i + (self.window_size // 2)  

                if window_end - i < self.window_size:
                    mid_index = i  

                if mid_index < total_elements:
                    binary_flat[mid_index] = 1 - binary_flat[mid_index]

            if bit_idx < total_m_bits:
                bit_idx += 1

        return binary_flat


    def __ReMappingToGaussianNoise(self, binary_matrix:np.ndarray) -> np.ndarray:
        Z_T_array = self.initialNoise.clone().cpu().numpy()
        for ch in range(self.channel):
            for i in range(self.height_blocks):
                for j in range(self.width_blocks):
                    original_cdf_value = norm.cdf(self.initialNoise_cpu[self.fix_batchsize-1,ch, i, j])
                    binary_value = binary_matrix[ch, i, j]
                    original_binary_value=original_cdf_value > 0.5
                    
                    if binary_value != original_binary_value:
                        if self.use_seed == 0:
                            u = np.random.uniform(0, 1)
                        else:
                            u = self.rng.uniform(0, 1)
                        Z_T_array[self.fix_batchsize-1,ch, i, j] = norm.ppf((u + binary_value) / 2**1)

        return Z_T_array


    def __LogINfo(self):
        self.LogInfo('WatermarkEmbed', 'Key_hex', self.key.hex())
        self.LogInfo('WatermarkEmbed', 'Oonce_hex',self.nonce.hex())
        self.LogInfo('WatermarkEmbed', 'Message',self.message)
        self.LogInfo('WatermarkEmbed', 'Setting message_length', self.message_length)
        self.LogInfo('WatermarkEmbed', 'RandomSeed',self.random_seed)


    def GetWatermarkedNoise(self, 
                       initial_noise: torch.Tensor, 
                       message: str,
                       message_length: int=-1, 
                       window_size: int=1, 
                       fix_batchsize: int=1
                       ) -> (torch.Tensor,list):

        self.initialNoise = initial_noise
        self.message = message
        self.message_length=message_length
        self.fix_batchsize = fix_batchsize
        self.channel = initial_noise.shape[1]
        self.window_size = window_size
    

        self.width_blocks = initial_noise.shape[-1] #image_width // 8
        self.height_blocks = initial_noise.shape[-2] #image_height // 8
        total_blocks_needed = self.channel * self.width_blocks * self.height_blocks // window_size

        watermark_bytes = self.__initMessage(total_blocks_needed, message_length)

        m_bits = self.__EncryptMessage(watermark_bytes)

        binary_flat = self.__NoiseToBinary()

        binary_flat = self.__CoreMethod(binary_flat, m_bits, total_blocks_needed)

        binary_matrix = binary_flat.reshape((self.channel, self.height_blocks, self.width_blocks))

        Z_T_array = self.__ReMappingToGaussianNoise(binary_matrix)

        self.__LogINfo()
    

        return torch.tensor(Z_T_array[0], dtype=torch.float32, device=self.device),self.ParamsNeedReturn

    

class DPRW_WatermarkExtract(Common):
    def __init__(self, key_hex:str=None, nonce_hex:str=None, device:str ='cpu', logger_file_path:str='./logs'):
        super().__init__(device, logger_file_path)
        self.key = bytes.fromhex(key_hex)
        self.nonce = bytes.fromhex(nonce_hex)

    
    def GetMessageFromLatents(self, 
                             reversed_latents: torch.Tensor, 
                             message_length: int, 
                             window_size: int) -> str:

        cipher = Cipher(algorithms.ChaCha20(self.key, self.nonce), mode=None, backend=default_backend())
        decryptor = cipher.decryptor()

        binary_matrix = (norm.cdf(reversed_latents.cpu().numpy()) > 0.5).astype(int)
        binary_flat = binary_matrix.flatten()
        total_elements = len(binary_flat)

        reconstructed_bits = []
        for i in range(0, total_elements, window_size):
            window_end = min(i + window_size, total_elements)
            window_sum = np.sum(binary_flat[i:window_end])
            y_reconstructed = 1 if window_sum % 2 != 0 else 0
            reconstructed_bits.append(y_reconstructed)

        extracted_binary_str = ''.join(str(bit) for bit in reconstructed_bits)
        extracted_bytes = []
        for i in range(0, len(extracted_binary_str), 8):
            chunk = extracted_binary_str[i:i + 8]
            extracted_bytes.append(int(chunk, 2))
        decrypted_bytes = decryptor.update(bytes(extracted_bytes)) + decryptor.finalize()

        all_bits = ''.join(format(byte, '08b') for byte in decrypted_bytes)

        segment_count = len(all_bits) // message_length
        segments = [all_bits[i:i + message_length] 
                    for i in range(0, segment_count * message_length, message_length)]

        reconstructed_message_bin = ''
        for i in range(message_length):
            count_1 = sum(segment[i] == '1' for segment in segments)
            if count_1 > len(segments) / 2:
                reconstructed_message_bin += '1'
            else:
                reconstructed_message_bin += '0'

        try:
            extracted_message = bytes(int(reconstructed_message_bin[i:i+8], 2) 
                                      for i in range(0, len(reconstructed_message_bin), 8)).decode('utf-8', errors='replace')
        except ValueError:
            extracted_message = "<Decoding Error>"

        self.LogInfo("WatermarkExtract",f"Reconstructed_message_bin",reconstructed_message_bin)
        self.LogInfo("WatermarkExtract",f"Extracted_message",extracted_message)

        return reconstructed_message_bin, extracted_message


class DPRW_WatermarkAccuracy(Common):
    def __init__(self, logger_file_path:str='./logs'):
        super().__init__(logger_file_path)
    
    def calculate_bit_accuracy(self,original_message: str, extracted_message_bin: str) -> (str, str, float):
        original_message_hex = original_message.encode('utf-8').hex()
        original_message_bin = bin(int(original_message_hex, 16))[2:].zfill(len(original_message_hex) * 4)
        min_length = min(len(original_message_bin), len(extracted_message_bin))
        original_message_bin, extracted_bin = original_message_bin[:min_length], extracted_message_bin[:min_length]

        matching_bits = sum(x == y for x, y in zip(original_message_bin, extracted_bin))
        bit_accuracy = matching_bits / min_length if min_length else 0.0

        try:
            extracted_message = bytes(int(extracted_bin[i:i+8], 2) 
                                      for i in range(0, len(extracted_bin), 8)).decode('utf-8', errors='replace')
        except ValueError:
            extracted_message = "<Decoding Error>"

        self.LogInfo("WatermarkAccuracy",f"Original_message_bin  ",original_message_bin)
        self.LogInfo("WatermarkAccuracy",f"Extracted_message_bin ",extracted_bin)
        self.LogInfo("WatermarkAccuracy",f"Bit_accuracy",bit_accuracy)   

        return bit_accuracy


