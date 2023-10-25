import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.doodle(
     image="https://img.freepik.com/premium-photo/close-up-dogs-face-with-big-smile-generative-ai_900101-62851.jpg",
     prompt="A cute dog",
)

base64_to_image(res.image_file).save("./doodle.png")
