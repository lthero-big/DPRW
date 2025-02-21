from DPRW_Watermark import DPRW_WatermarkEmbed
from DPRW_Watermark import DPRW_WatermarkExtract
from DPRW_Watermark import DPRW_WatermarkAccuracy



def main():
    config = {
        'image_height': 1024,
        'image_width': 1024,
        'channel': 4,
        'message': "lthero",
        'message_length': 256,
        'window_size': 1,
        'random_seed': 42,
        'use_seed': True,
        'device': 'cpu',
        'log_dir': './logs',
        "key_hex":"5822ff9cce6772f714192f43863f6bad1bf54b78326973897e6b66c3186b77a7",
        "nonce_hex":"05072fd1c2265f6f2e2a4080a2bfbdd8"
    }

    noise = torch.randn(1, config['channel'], config['image_height'] // 8, config['image_width'] // 8)
    embedder = DPRW_WatermarkEmbed(use_seed=config['use_seed'], key_hex=config['key_hex'] ,nonce_hex=config['nonce_hex'],random_seed=config['random_seed'], device=config['device'])
    watermarked_noise, params = embedder.embed(noise, config['message'], config['message_length'], config['window_size'])
    key_hex, nonce_hex = params[0].hex(), params[1].hex()

    extractor = DPRW_WatermarkExtract(key_hex, nonce_hex, config['device'])
    extracted_bin, extracted_msg = extractor.extract(watermarked_noise, config['message_length'], config['window_size'])

    accuracy = DPRW_WatermarkAccuracy().evaluate(config['message'], extracted_bin)

if __name__ == "__main__":
    main()
