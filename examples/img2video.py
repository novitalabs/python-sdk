import os

from novita_client import NovitaClient
from novita_client.utils import base64_to_image

client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URNOVITA_API_URII', None))
res = client.img2video(
    model_name="SVD-XT",
    steps=30,
    frames_num=25,
    image="https://replicate.delivery/pbxt/JvLi9smWKKDfQpylBYosqQRfPKZPntuAziesp0VuPjidq61n/rocket.png",
    enable_frame_interpolation=True
)


with open("test.mp4", "wb") as f:
    f.write(res.video_bytes[0])
