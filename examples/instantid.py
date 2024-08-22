
import os
from novita_client import NovitaClient, InstantIDControlnetUnit
import base64



if __name__ == '__main__':
	client = NovitaClient(os.getenv('NOVITA_API_KEY'), os.getenv('NOVITA_API_URI', None))

	res = client.instant_id(
		model_name="sdxlUnstableDiffusers_v8HEAVENSWRATH_133813.safetensors",
		face_images=[
			"https://raw.githubusercontent.com/InstantID/InstantID/main/examples/yann-lecun_resize.jpg",
		],
		prompt="Flat illustration, a Chinese a man, ancient style, wearing a red cloth, smile face, white skin, clean background, fireworks blooming, red lanterns",
		negative_prompt="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
		id_strength=0.8,
		adapter_strength=0.8,
		steps=20,
		seed=42,
		width=1024,
		height=1024,
		controlnets=[
			InstantIDControlnetUnit(
				model_name='controlnet-openpose-sdxl-1.0',
				strength=0.4,
				preprocessor='openpose',
			),
			InstantIDControlnetUnit(
				model_name='controlnet-canny-sdxl-1.0',
				strength=0.3,
				preprocessor='canny',
			),
		],
		response_image_type='jpeg',
		enterprise_plan=False,
	)

	print('res:', res)

	if hasattr(res, 'images_encoded'):
		with open(f"instantid.png", "wb") as f:
			f.write(base64.b64decode(res.images_encoded[0]))
