import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.replace_background(
    image="./telegram-cloud-photo-size-2-5408823814353177899-y.jpg",
    prompt="in living room, Christmas tree",
)
base64_to_image(res.image_file).save("./replace_background.png")
