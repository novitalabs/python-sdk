import os
import base64
from novita_client import NovitaClient

if __name__ == '__main__':
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
    res = client.instant_style(
            model_name = "protovisionXLHighFidelity3D_release0620Bakedvae_131308.safetensors",
            ref_image = "https://raw.githubusercontent.com/InstantID/InstantID/main/examples/yann-lecun_resize.jpg",
            style_mode = 2,
            prompt = "a man, masterpiece, best quality, high quality",
            negative_prompt = "bad quality, bad anatomy, worst quality, low quality, lowres, extra fingers, blur, blurry, ugly, wrong proportions, watermark, image artifacts, bad eyes, bad hands, bad arms",
            width = 768,
            height = 1024,
            image_num = 1,
            steps = 25,
            seed = -1,
            guidance_scale = 4.5,
            sampler_name = "Euler a",
            enterprise_plan = False,
    )
    print('res:', res)

    if hasattr(res, 'images_encoded'):
        with open(f"instantstyle.png", "wb") as f:
            f.write(base64.b64decode(res.images_encoded[0]))