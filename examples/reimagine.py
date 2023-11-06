import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.reimagine(
    image="/home/anyisalin/develop/novita-client-python/examples/doodle-generated.png"
)

base64_to_image(res.image_file).save("./reimagine.png")
