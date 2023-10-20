import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'))
res = client.reimagine(
    image="https://madera.objects.liquidweb.services/photos/20371-yosemite-may-yosemite-falls-waterfalls-cooks-meadow-spring-2023-Rectangle-600x400.jpg"
)

base64_to_image(res.image_file).save("./reimagine.png")
