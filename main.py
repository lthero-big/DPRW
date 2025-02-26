from DPRW_Watermark import DPRW_WatermarkEmbed
from DPRW_Watermark import DPRW_WatermarkExtract
from DPRW_Watermark import DPRW_WatermarkAccuracy



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
        dtype=torch.float32,
        device="cuda"  
    )
    fix_batchsize = 1
    message = "5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7"
    width, height = 1024, 1024
    message_length = len(message)*8
    window_size = 1
    num_steps = 30
    generated_image_path = "output/generated_image.png"

    prompt = "One girl in a white sweater at Times Square"
    engine.generate_image(
        prompt=prompt, width=width, height=height, num_steps=num_steps,
        message=message, message_length=message_length, window_size=window_size,fix_batchsize=1,
        output_path=generated_image_path
    )

    reversed_latents = engine.invert_image(generated_image_path, width=width, 
                                           height=height, 
                                           num_steps=30, 
                                           guidance_scale=7.5)
    extracted_bin, extracted_msg = engine.extract_watermark(reversed_latents, 
                                                            message_length=message_length, 
                                                            window_size=window_size)
    accuracy = engine.evaluate_accuracy(message, extracted_bin,extracted_msg)
if __name__ == "__main__":
    example_usage()
