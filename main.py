from DPRW_Watermark import DPRW_WatermarkEmbed
from DPRW_Watermark import DPRW_WatermarkExtract
from DPRW_Watermark import DPRW_WatermarkAccuracy



def main():
    Image_height = 512
    Image_width = 512
    channel=4
    message = "lthero"
    message_length=64
    window_size = 1
    random_seed = 42
    use_seed = 1 if random_seed!=-1 else 0
    sample_noise = torch.randn((1, channel, Image_height//8, Image_width//8)) 
    
    # key_hex="5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"
    # nonce_hex="05072fd1c2265f6f2e2a4080a2bfbdd8"
    watermark_embed = DPRW_WatermarkEmbed(
        key_hex="this key_hex is not valid",
        nonce_hex="this nonce_hex is not valid",
        device="cpu",
        use_seed=use_seed,
        random_seed=random_seed
    )

    watermarked_noise,params = watermark_embed.GetWatermarkedNoise(
        initial_noise=sample_noise,
        message=message,
        message_length=message_length,  
        window_size=window_size
    )
    # if params is not empty means the key_hex and nonce_hex need to use the params
    if len(params)>0:
        key_hex=params[0].hex()
        nonce_hex=params[1].hex()

    # Assume the watermarked_noise is the reversed_latents, but use the real reversed_latents if it's not the case
    reversed_latents = watermarked_noise

    watermark_extract = DPRW_WatermarkExtract(
        key_hex=key_hex,
        nonce_hex=nonce_hex,
        device="cpu",
    )

    extracted_bin,extracted_message = watermark_extract.GetMessageFromLatents(
        reversed_latents=reversed_latents,
        message_length=message_length,
        window_size=window_size
    )

    watermark_accuracy = DPRW_WatermarkAccuracy(
        logger_file_path="./logs"
    )

    accuracy = watermark_accuracy.calculate_bit_accuracy(
        original_message=message,
        extracted_message_bin=extracted_bin
    )


if __name__ == "__main__":
    main()
