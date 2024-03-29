from novita_client import *
from novita_client.utils import save_image, read_image_to_base64, base64_to_image
import os
from PIL import Image
import random
import sys


def test_lcm_txt2img():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

    x, y = 0, 0
    background = Image.new('RGB', (512 * 10, 512 * 10), (255, 255, 255))
    for i in range(20):
        animals = ["cat", "dog", "bird", "horse", "elephant", "giraffe", "zebra", "lion", "tiger", "bear", "sheep", "cow", "pig"]
        res = client.lcm_txt2img(
            prompt=f"a cute {random.choice(animals)}, masterpiece, best quality, realism",
            steps=8,
            image_num=5,
        )
        images = [base64_to_image(img.image_file) for img in res.images]
        for image in images:
            background.paste(image, (x, y))
            background.save("lcm.jpeg")
            x += 512
            if x >= 512 * 10:
                x = 0
                y += 512


def test_normal_txt2img():
    client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

    x, y = 0, 0
    background = Image.new('RGB', (512 * 10, 512 * 10), (255, 255, 255))
    for i in range(20):
        animals = ["cat", "dog", "bird", "horse", "elephant", "giraffe", "zebra", "lion", "tiger", "bear", "sheep", "cow", "pig"]
        res = client.sync_txt2img(
            Txt2ImgRequest(
                prompt=f"a cute {random.choice(animals)}, masterpiece, best quality, realism",
                steps=20,
                height=512,
                width=512,
                batch_size=5,
            )
        )
        images = [Image.open(BytesIO(b) for b in res.data.imgs_bytes)]
        for image in images:
            background.paste(image, (x, y))
            background.save("normal.jpeg")
            x += 512 
            if x >= 512 * 10:  
                x = 0
                y += 512


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "normal":
            test_normal_txt2img()
        else:
            test_lcm_txt2img()
