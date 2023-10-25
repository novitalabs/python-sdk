import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))
res = client.remove_text(
    image="https://images.uiiiuiii.com/wp-content/uploads/2023/07/i-banner-20230714-1.jpg"
)

base64_to_image(res.image_file).save("./remove_text.png")
