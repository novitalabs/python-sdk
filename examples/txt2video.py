import os

from novita_client import NovitaClient
from novita_client.utils import save_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.txt2video(
        model_name = "dreamshaper_8_93211.safetensors",
        prompts = [{
                    "prompt": "A girl, baby, portrait, 5 years old",
                    "frames": 16,},
                    {
                    "prompt": "A girl, child, portrait, 10 years old",
                    "frames": 16,
                    }
                    ],
        steps = 20,
        guidance_scale = 10,
        height = 512,
        width = 768,
        clip_skip = 4,
        negative_prompt = "a rainy day",
        response_video_type = "mp4",
    )
save_image(res.video_bytes[0], 'output.mp4')
