import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.create_tile(
    prompt="a cute flower",
)


base64_to_image(res.image_file).save("./create-tile.png")
