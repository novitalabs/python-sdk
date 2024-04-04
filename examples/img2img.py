import pdb
import os

from novita_client import NovitaClient, Img2ImgV3ControlNetUnit, ControlNetPreprocessor, Img2ImgV3Embedding
from novita_client.utils import base64_to_image, input_image_to_pil

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.img2img_v3(
    model_name="MeinaHentai_V5.safetensors",
    steps=30,
    height=512,
    width=512,
    input_image="https://img.freepik.com/premium-photo/close-up-dogs-face-with-big-smile-generative-ai_900101-62851.jpg",
    prompt="1 cute dog",
    strength=0.5,
    guidance_scale=7,
    embeddings=[Img2ImgV3Embedding(model_name=_) for _ in [
        "bad-image-v2-39000",
        "verybadimagenegative_v1.3_21434",
        "BadDream_53202",
        "badhandv4_16755",
        "easynegative_8955.safetensors"]],
    seed=-1,
    sampler_name="DPM++ 2M Karras",
    clip_skip=2,
    # controlnet_units=[Img2ImgV3ControlNetUnit(
    #     model_name="control_v11f1p_sd15_depth",
    #     preprocessor="depth",
    #     image_base64="./20240309-003206.jpeg",
    #     strength=1.0
    # )]
)

base64_to_image(res.images_encoded[0]).save("./img2img.png")
