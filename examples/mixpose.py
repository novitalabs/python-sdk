import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.mixpose(
    image="https://image.uniqlo.com/UQ/ST3/my/imagesgoods/455359/item/mygoods_23_455359.jpg?width=494",
    pose_image="https://image.uniqlo.com/UQ/ST3/ca/imagesgoods/455359/item/cagoods_02_455359.jpg?width=494",
)

base64_to_image(res.image_file).save("./mixpose.png")
