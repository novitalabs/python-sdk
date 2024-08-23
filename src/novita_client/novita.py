#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging

from time import sleep

from .version import __version__

from .exceptions import *
from .proto import *

import requests
from . import settings
from .utils import input_image_to_base64, input_image_to_pil
from io import BytesIO
from multiprocessing.pool import ThreadPool


logger = logging.getLogger(__name__)


class NovitaClient:
    """NovitaClient is the main entry point for interacting with the Novita API."""

    def __init__(self, api_key, base_url=None):
        self.base_url = base_url
        if self.base_url is None:
            self.base_url = "https://api.novita.ai"
        self.api_key = api_key
        self.session = requests.Session()

        if not self.api_key:
            raise ValueError("NOVITA_API_KEY environment variable not set")

        # eg: {"all": [proto.ModelInfo], "checkpoint": [proto.ModelInfo], "lora": [proto.ModelInfo]}
        self._model_list_cache = None
        self._model_list_cache_v3 = None
        self._extra_headers = {}
        self._default_response_image_type = "jpeg"

    def set_response_image_type(self, image_type: str):
        self._default_response_image_type = image_type

    def set_extra_headers(self, headers: dict):
        self._extra_headers = headers

    def _get(self, api_path, params=None) -> dict:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': "novita-python-sdk/{}".format(__version__),
            'Accept-Encoding': 'gzip, deflate',
        }
        headers.update(self._extra_headers)

        logger.debug(f"[GET] params: {params}")

        response = self.session.get(
            self.base_url + api_path,
            headers=headers,
            params=params,
            timeout=settings.DEFAULT_REQUEST_TIMEOUT,
        )

        logger.debug(f"[GET] {self.base_url + api_path}, headers: {headers} response: {response.content}")
        if response.status_code != 200:
            logger.error(f"Request failed: {response}")
            raise NovitaResponseError(
                f"Request failed with status {response.status_code}")

        return response.json()

    def _post(self, api_path, data) -> dict:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': "novita-python-sdk/{}".format(__version__),
            'Accept-Encoding': 'gzip, deflate',
        }
        headers.update(self._extra_headers)

        logger.debug(f"[POST] {self.base_url + api_path}, headers: {headers} data: {data}")

        response = self.session.post(
            self.base_url + api_path,
            headers=headers,
            json=data,
            timeout=settings.DEFAULT_REQUEST_TIMEOUT,
        )

        logger.debug(f"[POST] response: {response.content}")
        if response.status_code != 200:
            logger.error(f"Request failed: {response}")
            raise NovitaResponseError(
                f"Request failed with status {response.status_code}, {response.content}")

        return response.json()

    def txt2img(self, request: Txt2ImgRequest, enterprise_plan: bool=False) -> Txt2ImgResponse:
        """Asynchronously generate images from request

        Args:
            request (Txt2ImgRequest): The request object containing the text and image generation parameters.

        Returns:
            Txt2ImgResponse: The response object containing the task ID and status URL.
        """
        response = self._post('/v2/txt2img' if not enterprise_plan else '/v2/enterprise/txt2img', request.to_dict())

        return Txt2ImgResponse.from_dict(response)

    def progress(self, task_id: str, enterprise_plan: bool=None) -> ProgressResponse:
        """Get the progress of a task.

        Args:
            task_id (str): The ID of the task to get the progress for.

        Returns:
            ProgressResponse: The response object containing the progress information for the task.
        """
        response = self._get('/v2/progress' if not enterprise_plan else '/v2/enterprise/progress', {
            'task_id': task_id,
        })

        return ProgressResponse.from_dict(response)

    def async_task_result(self, task_id: str) -> V3TaskResponse:
        response = self._get('/v3/async/task-result', {
            'task_id': task_id,
        })
        return V3TaskResponse.from_dict(response)

    def img2img(self, request: Img2ImgRequest, enterprise_plan: bool=False) -> Img2ImgResponse:
        """Asynchronously generate images from request

        Args:
            request (Img2ImgRequest): The request object containing the image and image generation parameters.

        Returns:
            Img2ImgResponse: The response object containing the task ID and status URL.
        """
        response = self._post('/v2/img2img' if not enterprise_plan else '/v2/enterprise/img2img', request.to_dict())

        return Img2ImgResponse.from_dict(response)

    def wait_for_task(self, task_id, wait_for: int = 300, callback: callable = None, enterprise_plan: bool=False) -> ProgressResponse:
        """Wait for a task to complete

        This method waits for a task to complete by periodically checking its progress. If the task is not completed within the specified time, an NovitaTimeoutError is raised.

        Args:
            task_id (_type_): The ID of the task to wait for.
            wait_for (int, optional): The maximum time to wait for the task to complete, in seconds. Defaults to 300.

        Raises:
            NovitaTimeoutError: If the task fails to complete within the specified time.

        Returns:
            ProgressResponse: The response object containing the progress information for the task.
        """
        i = 0

        while i < wait_for:
            logger.info(f"Waiting for task {task_id} to complete")

            progress = self.progress(task_id, enterprise_plan=enterprise_plan)

            if callback and callable(callback):
                try:
                    callback(progress)
                except Exception as e:
                    logger.error(f"Task {task_id} progress callback failed: {e}")

            logger.info(
                f"Task {task_id} progress eta_relative: {progress.data.eta_relative}")

            if progress.data.status.finished():
                logger.info(f"Task {task_id} completed")
                return progress

            sleep(settings.DEFAULT_POLL_INTERVAL)
            i += 1

        raise NovitaTimeoutError(
            f"Task {task_id} failed to complete in {wait_for} seconds")

    def wait_for_task_v3(self, task_id, wait_for: int = 300, callback: callable = None) -> V3TaskResponse:
        i = 0

        while i < wait_for:
            logger.info(f"Waiting for task {task_id} to complete")

            progress = self.async_task_result(task_id)

            if callback and callable(callback):
                try:
                    callback(progress)
                except Exception as e:
                    logger.error(f"Task {task_id} progress callback failed: {e}")

            if progress.finished():
                logger.info(f"Task {task_id} completed")
                logging.debug(f"Task {progress.task.task_type}/{progress.task.task_id} debug_info: {progress.extra.debug_info}")
                return progress

            sleep(settings.DEFAULT_POLL_INTERVAL)
            i += 1
        raise NovitaTimeoutError(
            f"Task {task_id} failed to complete in {wait_for} seconds")

    def sync_txt2img(self, request: Txt2ImgRequest, download_images=True, callback: callable = None, enterprise_plan: bool=False) -> ProgressResponse:
        """Synchronously generate images from request, optionally download images

        This method generates images synchronously from the given request object. If download_images is set to True, the generated images will be downloaded.

        Args:
            request (Txt2ImgRequest): The request object containing the input text and other parameters.
            download_images (bool, optional): Whether to download the generated images. Defaults to True.

        Raises:
            NovitaResponseError: If the text to image generation fails.

        Returns:
            ProgressResponse: The response object containing the task status and generated images.
        """
        response = self.txt2img(request, enterprise_plan=enterprise_plan)

        if response.data is None:
            raise NovitaResponseError(f"Text to Image generation failed with response {response.msg}, code: {response.code}")

        res = self.wait_for_task(response.data.task_id, callback=callback, enterprise_plan=enterprise_plan)
        if download_images:
            res.download_images()
        return res

    def sync_img2img(self, request: Img2ImgRequest, download_images=True, callback: callable = None, enterprise_plan: bool=False) -> ProgressResponse:
        """Synchronously generate images from request, optionally download images

        Args:
            request (Img2ImgRequest): The request object containing the input image and other parameters.
            download_images (bool, optional): Whether to download the generated images. Defaults to True.

        Returns:
            ProgressResponse: The response object containing the task status and generated images.
        """
        response = self.img2img(request, enterprise_plan=enterprise_plan)

        if response.data is None:
            raise NovitaResponseError(f"Image to Image generation failed with response {response.msg}, code: {response.code}")

        res = self.wait_for_task(response.data.task_id, callback=callback, enterprise_plan=enterprise_plan)
        if download_images:
            res.download_images()
        return res

    def sync_upscale(self, request: UpscaleRequest, download_images=True, callback: callable = None, enterprise_plan: bool=False) -> ProgressResponse:
        """Syncronously upscale image from request, optionally download images

        Args:
            request (UpscaleRequest): _description_
            download_images (bool, optional): _description_. Defaults to True.

        Returns:
            ProgressResponse: _description_
        """
        response = self.upscale(request, enterprise_plan=enterprise_plan)

        if response.data is None:
            raise NovitaResponseError(f"Upscale failed with response {response.msg}, code: {response.code}")

        res = self.wait_for_task(response.data.task_id, callback=callback, enterprise_plan=enterprise_plan)
        if download_images:
            res.download_images()
        return res

    def upscale(self, request: UpscaleRequest) -> UpscaleResponse:
        """Upscale image

        This method sends a request to the Novita API to upscale an image using the specified parameters.

        Args:
            request (UpscaleRequest): An object containing the input image and other parameters.

        Returns:
            UpscaleResponse: An object containing the task status and the URL of the upscaled image.
        """
        response = self._post('/v2/upscale', request.to_dict())

        return UpscaleResponse.from_dict(response)

    def raw_adetailer(self, req: ADETailerRequest, extra: CommonV3Extra = None) -> ADETailerResponse:
        _req = CommonV3Request(request=req, extra=extra)

        return ADETailerResponse.from_dict(self._post('/v3/async/adetailer', _req.to_dict()))

    def adetailer(self, model_name: str, input_images: List[InputImage], prompt: str, sampler_name=None, guidance_scale=None, steps=None, strength=None, loras: List[ADETailerLoRA] = None, embeddings: List[ADETailerEmbedding] = None, negative_prompt=None, sd_vae=None, seed=None, clip_skip=None,  download_images=True, callback: callable = None,**kwargs) -> V3TaskResponse:
        req = ADETailerRequest(
            model_name=model_name,
            prompt=prompt,
        )
        if steps is not None:
            req.steps = steps
        if strength is not None:
            req.strength = strength
        if negative_prompt is not None:
            req.negative_prompt = negative_prompt
        if sd_vae is not None:
            req.sd_vae = sd_vae
        if seed is not None:
            req.seed = seed
        if clip_skip is not None:
            req.clip_skip = clip_skip
        if loras is not None:
            req.loras = loras
        if embeddings is not None:
            req.embeddings = embeddings
        if guidance_scale is not None:
            req.guidance_scale = guidance_scale
        if sampler_name is not None:
            req.sampler_name = sampler_name

        mode = "assets"  # or assets
        for input_image in input_images:
            if isinstance(input_image, str) and input_image.startswith("https://faas-output-image"):
                mode = "s3_url"
                break

        if mode == "assets":
            req.image_assets_ids = self.upload_assets([input_image_to_base64(image) for image in input_images])
        else:
            req.image_urls = input_images
        
        extra = CommonV3Extra(**kwargs)

        res = self.raw_adetailer(req, extra)

        return self.wait_for_task_v3(res.task_id, callback=callback)

        # def adetailer(self, model_name: str, image: InputImage, prompt: str, steps=None, strength=None, negative_prompt=None, vae=None, seed=None, clip_skip=None,  download_images=True, callback: callable = None) -> ProgressResponse:
        #     response = self.async_adetailer(model_name, image, prompt, steps, strength, negative_prompt, vae, seed, clip_skip)
        #     if response.data is None:
        #         raise NovitaResponseError(f"Upscale failed with response {response.msg}, code: {response.code}")

        #     res = self.wait_for_task(response.data.task_id, callback=callback)
        #     if download_images:
        #         res.download_images()
        #     return res

        # def async_adetailer(self, model_name: str, image: InputImage, prompt: str, steps=None, strength=None, negative_prompt=None, vae=None, seed=None, clip_skip=None) -> ProgressResponse:
        #     image_b64 = input_image_to_base64(image)
        #     request = ADETailerRequest(
        #         model_name=model_name,
        #         input_image=image_b64,
        #         prompt=prompt,
        #     )
        #     if steps is not None:
        #         request.steps = steps
        #     if strength is not None:
        #         request.strength = strength
        #     if negative_prompt is not None:
        #         request.negative_prompt = negative_prompt
        #     if vae is not None:
        #         request.vae = vae
        #     if seed is not None:
        #         request.seed = seed
        #     if clip_skip is not None:
        #         request.clip_skip = clip_skip
        #     return ADETailerResponse.from_dict(self._post('/v2/adetailer', request.to_dict()))

    def cleanup(self, image: InputImage, mask: InputImage, response_image_type=None, enterprise_plan: bool=None) -> CleanupResponse:
        image_b64 = input_image_to_base64(image)
        mask_b64 = input_image_to_base64(mask)
        request = CleanupRequest(image_file=image_b64, mask_file=mask_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        

        return CleanupResponse.from_dict(self._post('/v3/cleanup', request.to_dict()))

    def outpainting(self, image: InputImage, width=None, height=None, center_x=None, center_y=None, response_image_type=None, enterprise_plan: bool=None) -> OutpaintingResponse:
        image_b64 = input_image_to_base64(image)
        request = OutpaintingRequest(image_file=image_b64)
        if width is not None:
            request.width = width
        if height is not None:
            request.height = height
        if center_x is not None:
            request.center_x = center_x
        if center_y is not None:
            request.center_y = center_y

        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)

        request.set_image_type(self._default_response_image_type)
        return OutpaintingResponse.from_dict(self._post('/v3/outpainting', request.to_dict()))

    def remove_background(self, image: InputImage, response_image_type=None, enterprise_plan: bool=None) -> RemoveBackgroundResponse:
        image_b64 = input_image_to_base64(image)
        request = RemoveBackgroundRequest(image_file=image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)

        return RemoveBackgroundResponse.from_dict(self._post('/v3/remove-background', request.to_dict()))

    def remove_text(self, image: InputImage, response_image_type=None, enterprise_plan: bool=None) -> RemoveTextResponse:
        image_b64 = input_image_to_base64(image)
        request = RemoveTextRequest(image_file=image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)

        return RemoveTextResponse.from_dict(self._post('/v3/remove-text', request.to_dict()))

    def reimagine(self, image: InputImage, response_image_type=None, enterprise_plan: bool=None) -> ReimagineResponse:
        image_b64 = input_image_to_base64(image)
        request = ReimagineRequest(image_file=image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)

        return ReimagineResponse.from_dict(self._post('/v3/reimagine', request.to_dict()))

    def doodle(self, image: InputImage, prompt: str, similarity: float = None, response_image_type=None, enterprise_plan: bool=None) -> DoodleResponse:
        image_b64 = input_image_to_base64(image)
        request = DoodleRequest(image_file=image_b64, prompt=prompt)
        if similarity is not None:
            request.similarity = similarity

        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)

        return DoodleResponse.from_dict(self._post('/v3/doodle', request.to_dict()))
    
    def relight(self, input_image: str, model_name: str,lighting_preference:str, prompt: str, steps:int, sampler_name:str,guidance_scale: float, strength:float,\
                seed:int=-1,background_image_file:str=None,negative_prompt:str=None,clip_skip:int=None,response_image_type:str=None, enterprise_plan: bool=None) -> RelightResponse:
        image_b64 = input_image_to_base64(input_image)
        request = RelightRequest(image_file=image_b64, model_name=model_name,prompt=prompt,lighting_preference=lighting_preference,steps=steps,sampler_name = sampler_name,seed =seed, background_image_file\
                                 = background_image_file,negative_prompt = negative_prompt,strength=strength,guidance_scale=guidance_scale)
        if (clip_skip is not None):
            request.clip_skip = clip_skip
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)

        return RelightResponse.from_dict(self._post('/v3/relight', request.to_dict()))

    def mixpose(self, image: InputImage, pose_image: InputImage, response_image_type=None, enterprise_plan: bool=None) -> MixPoseResponse:
        image_b64 = input_image_to_base64(image)
        pose_image_b64 = input_image_to_base64(pose_image)
        request = MixPoseRequest(image_file=image_b64, pose_image_file=pose_image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)

        return MixPoseResponse.from_dict(self._post('/v3/mix-pose', request.to_dict()))

    def replace_background(self, image: InputImage, prompt: str, response_image_type=None, enterprise_plan: bool=None) -> ReplaceBackgroundResponse:
        image_b64 = input_image_to_base64(image)
        request = ReplaceBackgroundRequest(image_file=image_b64, prompt=prompt)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return ReplaceBackgroundResponse.from_dict(self._post('/v3/replace-background', request.to_dict()))

    def replace_sky(self, image: InputImage, sky: str, response_image_type=None, enterprise_plan: bool=None) -> ReplaceSkyResponse:
        image_b64 = input_image_to_base64(image)
        request = ReplaceSkyRequest(image_file=image_b64, sky=sky)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return ReplaceSkyResponse.from_dict(self._post('/v3/replace-sky', request.to_dict()))
    
    def remove_watermark(self, image: InputImage, response_image_type=None, enterprise_plan: bool=None) -> RemoveWatermarkResponse:
        image_b64 = input_image_to_base64(image)
        request = RemoveWatermarkRequest(image_file=image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return RemoveWatermarkResponse.from_dict(self._post('/v3/remove-watermark', request.to_dict()))

    def replace_object(self, image: InputImage, object_prompt: str, prompt: str, negative_prompt=None, response_image_type=None, enterprise_plan: bool=None) -> ReplaceObjectResponse:
        res: V3AsyncSubmitResponse = self.async_replace_object(image, object_prompt, prompt, negative_prompt, response_image_type, enterprise_plan)
        final_res = self.wait_for_task_v3(res.task_id)
        final_res.download_images()
        return ReplaceObjectResponse(
            image_file=final_res.images_encoded[0],
            image_type=final_res.images[0].image_type,
        )

    def async_replace_object(self, image: InputImage, object_prompt: str, prompt: str, negative_prompt=None, response_image_type=None, enterprise_plan: bool=None) -> ReplaceObjectResponse:
        image_b64 = input_image_to_base64(image)
        request = ReplaceObjectRequest(image_file=image_b64, object_prompt=object_prompt, prompt=prompt, negative_prompt=negative_prompt)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return V3AsyncSubmitResponse.from_dict(self._post('/v3/async/replace-object', request.to_dict()))

    def async_txt2video(self,model_name:str, height:int,width:int,steps:int,prompts:List[Txt2VideoPrompt],guidance_scale:float,seed:int=None,negative_prompt: Optional[str] = None,loras:List[Txt2VideoLoRA]=None,\
                        embeddings:List[Txt2VideoEmbedding]=None,clip_skip:int=None,closed_loop:bool=None,response_video_type:str=None,enterprise_plan: bool=None) -> Txt2VideoResponse:
        request = Txt2VideoRequest(model_name=model_name,height=height,width=width,steps=steps,prompts=prompts,negative_prompt=negative_prompt,guidance_scale=guidance_scale,loras=loras,embeddings=embeddings,clip_skip=clip_skip)
        if seed is not None:
            request.seed = seed
        if closed_loop is not None:
            request.closed_loop = closed_loop
        if response_video_type is not None:
            request.set_video_type(response_video_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return Txt2VideoResponse.from_dict(self._post('/v3/async/txt2video', request.to_dict()))
    
    def txt2video(self,model_name:str, height:int,width:int,steps:int,prompts:List[Txt2VideoPrompt],guidance_scale:float,negative_prompt: Optional[str] = None,seed:int=None,loras:List[Txt2VideoLoRA]=None,\
                        embeddings:List[Txt2VideoEmbedding]=None,clip_skip:int=None,closed_loop:bool=None,response_video_type:str=None,enterprise_plan: bool=None) -> Txt2VideoResponse:
        res: Txt2VideoResponse = self.async_txt2video(model_name, height, width, steps, prompts, guidance_scale, seed,negative_prompt, loras, embeddings, clip_skip, closed_loop,response_video_type,enterprise_plan)
        final_res = self.wait_for_task_v3(res.task_id)
        if final_res.task.status == V3TaskResponseStatus.TASK_STATUS_SUCCEED:
            final_res.download_videos()
        else:
            raise NovitaResponseError(f"")
        return final_res
    
    
    def async_img2video(self, image: InputImage, model_name: str, steps: int, frames_num: int, frames_per_second: int = 6, seed: int = None, image_file_resize_mode: str = Img2VideoResizeMode.CROP_TO_ASPECT_RATIO,\
                        motion_bucket_id: int = 127, cond_aug: float = 0.02, enable_frame_interpolation: bool = False, response_video_type:str=None,enterprise_plan: bool=None) -> Img2VideoResponse:
        image_b64 = input_image_to_base64(image)
        request = Img2VideoRequest(model_name=model_name, image_file=image_b64, steps=steps, frames_num=frames_num, frames_per_second=frames_per_second, seed=seed,
                                   image_file_resize_mode=image_file_resize_mode, motion_bucket_id=motion_bucket_id, cond_aug=cond_aug, enable_frame_interpolation=enable_frame_interpolation)
        if response_video_type is not None:
            request.set_video_type(response_video_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return Img2VideoResponse.from_dict(self._post('/v3/async/img2video', request.to_dict()))

    def img2video(self, image: InputImage, model_name: str, steps: int, frames_num: int, frames_per_second: int = 6, seed: int = None, image_file_resize_mode: str = Img2VideoResizeMode.CROP_TO_ASPECT_RATIO, motion_bucket_id: int = 127, cond_aug: float = 0.02, enable_frame_interpolation: bool = False, response_video_type: str=None, enterprise_plan: bool=None) -> Img2VideoResponse:
        res: Img2VideoResponse = self.async_img2video(image, model_name, steps, frames_num, frames_per_second, seed, image_file_resize_mode, motion_bucket_id, cond_aug, enable_frame_interpolation, response_video_type, enterprise_plan)
        final_res = self.wait_for_task_v3(res.task_id)
        final_res.download_videos()
        return final_res
    
    def async_img2video_motion(self,image_assets_id:str,motion_assets_id:str,seed:int=None,response_video_type:str=None,enterprise_plan: bool=None) -> Img2VideoMotionResponse:
        request = Img2VideoMotionRequest(image_assets_id=image_assets_id,motion_assets_id=motion_assets_id,seed=seed)
        if response_video_type is not None:
            request.set_video_type(response_video_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return Img2VideoMotionResponse.from_dict(self.post('/v3/async/img2video-motion',request.to_dict()))

    def img2video_motion(self,image_assets_id:str,motion_assets_id:str,seed:int=None,set_video_type:str=None,enterprise_plan: bool=None) -> Img2VideoMotionResponse:
        res: Img2VideoMotionResponse = self.async_img2video_motion(image_assets_id,motion_assets_id,seed,set_video_type,enterprise_plan)
        final_res = self.wait_for_task_v3(res.task_id)
        return final_res
    
    def async_animated_anyone(self, image_assets_id: str, pose_video_assets_id: str, height:int, width:int, steps:int, seed:int, set_video_type:str=None, enterprise_plan: bool=None) -> AnimatedAnyoneResponse:
        request = AnimatedAnyoneRequest(image_assets_id=image_assets_id, pose_video_assets_id=pose_video_assets_id, height=height, width=width, steps=steps, seed=seed)
        if set_video_type is not None:
            request.set_video_type(set_video_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return AnimatedAnyoneResponse.from_dict(self._post('/v3/async/animated-anyone', request.to_dict()))
    
    def animated_anyone(self, image_assets_id: str, pose_video_assets_id: str, height:int, width:int, steps:int, seed:int, set_video_type:str=None, enterprise_plan=None) -> AnimatedAnyoneResponse:
        res: AnimatedAnyoneResponse = self.async_animated_anyone(image_assets_id,pose_video_assets_id,height,width,steps,seed,set_video_type, enterprise_plan)
        final_res = self.wait_for_task_v3(res.task_id)
        return final_res


    def lcm_img2img(self, model_name: str, image: InputImage, prompt: str, image_num: int, negative_prompt: str = None, steps: int = None, guidance_scale: float = None, clip_skip: int = None, sd_vae: str = None, loras: List[LCMLoRA] = None, embeddings: List[LCMEmbedding] = None) -> LCMImg2ImgResponse:
        res = self._post('/v3/lcm-img2img', LCMImg2ImgRequest(
            input_image=input_image_to_base64(image),
            prompt=prompt,
            model_name=model_name,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            image_num=image_num,
            clip_skip=clip_skip,
            sd_vae=sd_vae,
            loras=loras,
            embeddings=embeddings,
        ).to_dict())
        return LCMImg2ImgResponse.from_dict(res)

    def restore_face(self, image: InputImage, fidelity=None, response_image_type=None, enterprise_plan=None) -> RestoreFaceResponse:
        image_b64 = input_image_to_base64(image)
        request = RestoreFaceRequest(image_file=image_b64)
        if fidelity is not None:
            request.fidelity = fidelity
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return RestoreFaceResponse.from_dict(self._post('/v3/restore-face', request.to_dict()))

    def create_tile(self, prompt: str, negative_prompt=None, width=None, height=None, response_image_type=None, enterprise_plan=None) -> CreateTileResponse:
        request = CreateTileRequest(prompt=prompt)
        if negative_prompt is not None:
            request.negative_prompt = negative_prompt
        if width is not None:
            request.width = width
        if height is not None:
            request.height = height
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return CreateTileResponse.from_dict(self._post('/v3/create-tile', request.to_dict()))
    

    def raw_inpainting(self, req: InpaintingRequest, extra: CommonV3Extra = None) -> InpaintingResponse:
        _req = CommonV3Request(request=req, extra=extra)
        return InpaintingResponse.from_dict(self._post('/v3/async/inpainting', _req.to_dict()))


    def async_inpainting(self, model_name: str, image: str, mask: str,\
                prompt: str,image_num: int,sampler_name: str, steps:int, guidance_scale: float,\
                seed: int, mask_blur: int=None, negative_prompt: str=None,\
                sd_vae:str=None,loras: List[InpaintingLoRA] = None,\
                embeddings: List[InpaintingEmbedding] = None,\
                clip_skip: int = None, strength: float = None,\
                inpainting_full_res: int=0, inpainting_full_res_padding: int=8,\
                inpainting_mask_invert: int=0, initial_noise_multiplier: float=0.5, **kwargs)\
                        -> InpaintingResponse:
        request = InpaintingRequest(model_name=model_name, image_base64=image, mask_image_base64=mask, \
                                    prompt=prompt,sampler_name=sampler_name, image_num=image_num, steps=steps,\
                                    guidance_scale=guidance_scale, seed=seed, mask_blur=mask_blur,loras= loras,embeddings=embeddings,\
                                    negative_prompt=negative_prompt, clip_skip=clip_skip, strength=strength,\
                                    inpainting_full_res=inpainting_full_res, inpainting_full_res_padding=inpainting_full_res_padding,\
                                    inpainting_mask_invert=inpainting_mask_invert, initial_noise_multiplier=initial_noise_multiplier)
        extra = CommonV3Extra(**kwargs)
        return self.raw_inpainting(request, extra)

    def inpainting(self, model_name: str, image: InputImage, mask: InputImage,\
                prompt: str,image_num: int,sampler_name: str, steps:int, guidance_scale: float,\
                seed: int, mask_blur: int=None, negative_prompt: str=None,\
                sd_vae:str=None,loras: List[InpaintingLoRA] = None,\
                embeddings: List[InpaintingEmbedding] = None,\
                clip_skip: int = None, strength: float = None,\
                inpainting_full_res: int=0, inpainting_full_res_padding: int=8,\
                inpainting_mask_invert: int=0, initial_noise_multiplier: float=0.5,**kwargs) -> InpaintingResponse:
        input_image = input_image_to_base64(image)
        mask_image = input_image_to_base64(mask)
        res: InpaintingResponse = self.async_inpainting(model_name, input_image, mask_image, prompt, image_num, sampler_name, steps, guidance_scale, seed, mask_blur, negative_prompt, sd_vae, loras, embeddings, clip_skip, strength, inpainting_full_res, inpainting_full_res_padding, inpainting_mask_invert, initial_noise_multiplier, **kwargs)
        final_res = self.wait_for_task_v3(res.task_id)
        if final_res.task.status == V3TaskResponseStatus.TASK_STATUS_SUCCEED:
            final_res.download_images()
        else:
            logging.error(f"Failed to inpaint image: {final_res.task.status}")
            raise NovitaResponseError(f"Task {final_res.task.task_id} failed with status {final_res.task.status}")
        return final_res
    
    def img2mask(self, image: InputImage,response_image_type=None,enterprise_plan=None) -> Img2MaskResponse:
        input_image = input_image_to_base64(image)
        resquest = Img2MaskRequest(image_file=input_image)
        if response_image_type is None:
            resquest.set_image_type(self._default_response_image_type)
        else:
            resquest.set_image_type(response_image_type)
        if enterprise_plan is not None:
            resquest.set_enterprise_plan(enterprise_plan)
        else:
            resquest.set_enterprise_plan(False)
        return Img2MaskResponse.from_dict(self._post('/v3/img2mask', resquest.to_dict()))

    def img2prompt(self, image: InputImage) -> Img2PromptResponse:
        input_image = input_image_to_base64(image)
        resquest = Img2PromptRequest(image_file=input_image)
        return Img2PromptResponse.from_dict(self._post('/v3/img2prompt', resquest.to_dict()))

    def merge_face(self, image: InputImage, face_image: InputImage, response_image_type=None, enterprise_plan=None) -> MergeFaceResponse:
        input_image = input_image_to_base64(image)
        face_image = input_image_to_base64(face_image)
        request = MergeFaceRequest(image_file=input_image, face_image_file=face_image)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        if enterprise_plan is not None:
            request.set_enterprise_plan(enterprise_plan)
        else:
            request.set_enterprise_plan(False)
        return MergeFaceResponse.from_dict(self._post('/v3/merge-face', request.to_dict()))

    def lcm_txt2img(self, prompt: str, width=None, height=None, steps=None, guidance_scale=None, image_num=None) -> LCMTxt2ImgResponse:
        req = LCMTxt2ImgRequest(prompt=prompt)
        if width is not None:
            req.width = width
        if height is not None:
            req.height = height
        if steps is not None:
            req.steps = steps
        if guidance_scale is not None:
            req.guidance_scale = guidance_scale
        if image_num is not None:
            req.image_num = image_num
        return LCMTxt2ImgResponse.from_dict(self._post('/v3/lcm-txt2img', req.to_dict()))

    def upload_training_assets(self, images: List[InputImage], batch_size=10) -> List[str]:
        def _upload_assets(image: InputImage) -> str:
            pil_image = input_image_to_pil(image)
            buff = BytesIO()
            if pil_image.format != "JPEG":
                pil_image = pil_image.convert("RGB")
                pil_image.save(buff, format="JPEG")
            else:
                pil_image.save(buff, format="JPEG")

            upload_res: UploadAssetResponse = UploadAssetResponse.from_dict(self._post("/v3/assets/training_dataset", UploadAssetRequest(file_extension="jpeg").to_dict()))
            res = requests.put(upload_res.upload_url, data=buff.getvalue(), headers={'Content-type': 'image/jpeg'})
            if res.status_code != 200:
                raise NovitaResponseError(f"Failed to upload image: {res.content}")
            return upload_res

        with ThreadPool(batch_size) as pool:
            results = pool.map(_upload_assets, images)
            ret = []
            try:
                for return_value in results:
                    ret.append(return_value.assets_id)
            except Exception as e:
                raise NovitaResponseError(f"Failed to upload image: {e}")
            return ret

        # for image in images:
        #     pil_image = input_image_to_pil(image)
        #     buff = BytesIO()
        #     if pil_image.format != "JPEG":
        #         pil_image = pil_image.convert("RGB")
        #         pil_image.save(buff, format="JPEG")
        #     else:
        #         pil_image.save(buff, format="JPEG")

        #     upload_res: UploadAssetResponse = UploadAssetResponse.from_dict(self._post("/v3/assets/training_dataset", UploadAssetRequest(file_extension="jpeg").to_dict()))
        #     res = requests.put(upload_res.upload_url, data=buff.getvalue(), headers={'Content-type': 'image/jpeg'})
        #     if res.status_code != 200:
        #         raise NovitaResponseError(f"Failed to upload image: {res.content}")
        #     ret.append(upload_res.assets_id)
        # return ret

    def create_training_style(self,
                              name,
                              base_model,
                              images: List[InputImage],
                              captions: List[str],
                              width: int = 512,
                              height: int = 512,
                              learning_rate: str = None,
                              seed: int = None,
                              lr_scheduler: str = None,
                              with_prior_preservation: bool = None,
                              prior_loss_weight: float = None,
                              lora_r: int = None,
                              lora_alpha: int = None,
                              max_train_steps: str = None,
                              lora_text_encoder_r: int = None,
                              lora_text_encoder_alpha: int = None,
                              components=None
                              ):
        if len(images) != len(captions):
            raise ValueError("images and captions must have the same length")

        assets = self.upload_training_assets(images)
        req = CreateTrainingStyleRequest(
            name=name,
            base_model=base_model,
            image_dataset_items=[TrainingStyleImageDatasetItem(assets_id=assets_id, caption=caption) for assets_id, caption in zip(assets, captions)],
            expert_setting=TrainingExpertSetting(
                max_train_steps=max_train_steps,
                learning_rate=learning_rate,
                seed=seed,
                lr_scheduler=lr_scheduler,
                with_prior_preservation=with_prior_preservation,
                prior_loss_weight=prior_loss_weight,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_text_encoder_r=lora_text_encoder_r,
                lora_text_encoder_alpha=lora_text_encoder_alpha,
            ),
            components=[_.to_dict() for _ in components] if components is not None else None,
            width=width,
            height=height,
        )
        res = CreateTrainingStyleResponse.from_dict(self._post("/v3/training/style", req.to_dict()))
        return res.task_id

    def create_training_subject(self, name,
                                base_model,
                                images: List[InputImage],
                                instance_prompt: str,
                                class_prompt: str,
                                width: int = 512,
                                height: int = 512,
                                learning_rate: str = None,
                                seed: int = None,
                                lr_scheduler: str = None,
                                with_prior_preservation: bool = None,
                                prior_loss_weight: float = None,
                                lora_r: int = None,
                                lora_alpha: int = None,
                                max_train_steps: str = None,
                                lora_text_encoder_r: int = None,
                                lora_text_encoder_alpha: int = None,
                                components=None) -> str:
        assets = self.upload_training_assets(images)
        req = CreateTrainingSubjectRequest(
            name=name,
            base_model=base_model,
            image_dataset_items=[TrainingImageDatasetItem(assets_id=assets_id) for assets_id in assets],
            expert_setting=TrainingExpertSetting(
                instance_prompt=instance_prompt,
                class_prompt=class_prompt,
                max_train_steps=max_train_steps,
                learning_rate=learning_rate,
                seed=seed,
                lr_scheduler=lr_scheduler,
                with_prior_preservation=with_prior_preservation,
                prior_loss_weight=prior_loss_weight,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_text_encoder_r=lora_text_encoder_r,
                lora_text_encoder_alpha=lora_text_encoder_alpha,
            ),
            components=[_.to_dict() for _ in components] if components is not None else None,
            width=width,
            height=height,
        )
        res = CreateTrainingSubjectResponse.from_dict(self._post("/v3/training/subject", req.to_dict()))
        return res.task_id

    def query_training_subject_status(self, task_id: str) -> QueryTrainingSubjectStatusResponse:
        return QueryTrainingSubjectStatusResponse.from_dict(self._get("/v3/training/subject", params={"task_id": task_id}))

    def list_training(self, task_type: str = None) -> TrainingTaskList:
        params = {}
        if task_type is not None:
            params["task_type"] = task_type

        return TrainingTaskList(TrainingTaskListResponse.from_dict(self._get("/v3/training", params=params)).tasks)

    def upload_assets(self, images: List[InputImage], batch_size=10) -> List[str]:
        buffs = []
        for image in images:
            if os.path.exists(image):
                pil_image = input_image_to_pil(image)
                buff = BytesIO()
                if pil_image.format != "JPEG":
                    pil_image = pil_image.convert("RGB")
                    pil_image.save(buff, format="JPEG")
                else:
                    pil_image.save(buff, format="JPEG")
                buffs.append(buff)
            elif image.startswith("http") or image.startswith("https"):
                buff = BytesIO(requests.get(image).content)
                buffs.append(buff)


        def _upload_asset(buff):
            attempt = 5
            while attempt > 0:
                upload_res = requests.put("https://assets.novitai.com/image", data=buff.getvalue(), headers={'Content-type': 'image/jpeg'})
                if upload_res.status_code < 400:
                    return upload_res.json()["assets_id"]
                attempt -= 1
            raise NovitaResponseError(f"Failed to upload image: {upload_res.content}")

        with ThreadPool(batch_size) as pool:
            results = pool.map(_upload_asset, buffs)
            ret = []
            try:
                for return_value in results:
                    ret.append(return_value)
            except Exception as e:
                raise NovitaResponseError(f"Failed to upload image: {e}")
            return ret

    def async_make_photo(self, images: List[InputImage], model_name: str, prompt: str, loras: List[MakePhotoLoRA] = None, height: int = None, width: int = None,  negative_prompt: str = None, steps: int = None, guidance_scale: float = None, image_num: int = None,\
                         clip_skip: int = None, seed: int = None, strength: float = None, sampler_name: str = None, response_image_type: str = None, crop_face: bool = None, enterprise_plan=None) -> MakePhotoResponse:
        assets = self.upload_assets(images)
        req = MakePhotoRequest(
            model_name=model_name,
            image_assets_ids=assets,
            prompt=prompt,
        )
        if loras is not None:
            req.loras = loras
        if height is not None:
            req.height = height
        if width is not None:
            req.width = width
        if negative_prompt is not None:
            req.negative_prompt = negative_prompt
        if steps is not None:
            req.steps = steps
        if guidance_scale is not None:
            req.guidance_scale = guidance_scale
        if image_num is not None:
            req.image_num = image_num
        if clip_skip is not None:
            req.clip_skip = clip_skip
        if seed is not None:
            req.seed = seed
        if strength is not None:
            req.strength = strength
        if sampler_name is not None:
            req.sampler_name = sampler_name
        if crop_face is not None:
            req.crop_face = crop_face

        if response_image_type is None:
            req.set_image_type(self._default_response_image_type)
        else:
            req.set_image_type(response_image_type)
        
        if enterprise_plan is not None:
            req.set_enterprise_plan(enterprise_plan)
        else:
            req.set_enterprise_plan(False)

        return MakePhotoResponse.from_dict(self._post('/v3/async/make-photo', req.to_dict()))

    def make_photo(self, images: List[InputImage], model_name: str, prompt: str, loras: List[MakePhotoLoRA] = None, height: int = None, width: int = None,  negative_prompt: str = None, steps: int = None, guidance_scale: float = None, image_num: int = None, clip_skip: int = None, seed: int = None,\
                strength: float = None, sampler_name: str = None, response_image_type: str = None, download_images: bool = True, callback: callable = None, enterprise_plan: bool = None) -> V3TaskResponse:
        res: MakePhotoResponse = self.async_make_photo(images, model_name, prompt, loras, height, width, negative_prompt, steps,
                                                       guidance_scale, image_num, clip_skip, seed, strength, sampler_name, response_image_type,enterprise_plan)
        final_res = self.wait_for_task_v3(res.task_id, callback=callback)
        if final_res.task.status != V3TaskResponseStatus.TASK_STATUS_SUCCEED:
            logger.error(f"Task {final_res.task.task_id} failed with status {final_res.task.status}")
        else:
            if download_images:
                final_res.download_images()
        return final_res

    def instant_id(self,
                   face_images: List[InputImage],
                   ref_images: List[InputImage] = None,
                   model_name: str = None,
                   prompt: str = None,
                   negative_prompt: str = None,
                   width: int = None,
                   height: int = None,  # if size arguments (width or height) is None, default size is equal to reference image size
                   id_strength: float = None,
                   adapter_strength: float = None,
                   steps: int = 20,
                   seed: int = -1,
                   guidance_scale: float = 5.,
                   sampler_name: str = 'Euler',
                   controlnets: List[InstantIDControlnetUnit] = None,
                   loras: List[InstantIDLora] = None,
                   response_image_type: str = None,
                   download_images: bool = True,
                   callback: callable = None,
                   enterprise_plan=None
                   ):
        #face_images = [input_image_to_pil(img) for img in face_images]
        #ref_images = ref_images and [input_image_to_pil(img) for img in ref_images]

        face_image_assets_ids = self.upload_assets(face_images)
        if ref_images is not None and len(ref_images) > 0:
            ref_image_assets_ids = self.upload_assets(ref_images)
        else:
            ref_image_assets_ids = face_image_assets_ids[:1]

        if width is None or height is None:
            ref_img = ref_images[0] if ref_images and len(ref_images) > 0 else face_images[0]
            width, height = ref_img.size

        payload_data = InstantIDRequest(
            face_image_assets_ids=face_image_assets_ids,
            ref_image_assets_ids=ref_image_assets_ids,
            model_name=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            id_strength=id_strength,
            adapter_strength=adapter_strength,
            steps=steps,
            seed=seed,
            guidance_scale=guidance_scale,
            sampler_name=sampler_name,
            controlnet=InstantIDRequestControlNet(units=controlnets),
            loras=loras,
        )

        if response_image_type is not None:
            payload_data.set_image_type(response_image_type)
        if enterprise_plan is not None:
            payload_data.set_enterprise_plan(enterprise_plan)
        else:
            payload_data.set_enterprise_plan(False)

        res = self._post("/v3/async/instant-id", payload_data.to_dict())
        final_res = self.wait_for_task_v3(res["task_id"], callback=callback)
        if final_res.task.status != V3TaskResponseStatus.TASK_STATUS_SUCCEED:
            logger.error(f"Task {final_res.task.task_id} failed with status {final_res.task.status}")
        else:
            if download_images:
                final_res.download_images()

        return final_res
    
    def async_instant_style(self,
                   ref_image: InputImage,
                   model_name: str,
                   prompt: str,
                   width: int,
                   height: int,
                   image_num: int,
                   steps: int,
                   guidance_scale: float,
                   sampler_name: str,
                   source_image: InputImage = None,
                   style_mode: int = None,
                   source_image_conditioning_scale: float = 0.5,
                   negative_prompt: str = None,
                   seed: str = -1,
                   clip_skip: int = None,
                   loras: List[InstantStyleLoRA] = None,
                   embeddings: List[InstantStyleEmbedding] = None,
                   enterprise_plan: bool = None,
                   response_image_type: str = None,
                   ):
        ref_image_assets_ids= self.upload_assets([ref_image])
        if source_image is not None:
            source_image_assets_ids = self.upload_assets([source_image])
        else:
            source_image_assets_ids = ref_image_assets_ids[:1]
        req = InstantStyleRequest(
            ref_image_assets_id=ref_image_assets_ids[0],
            source_image_assets_id=source_image_assets_ids[0],
            model_name=model_name,
            prompt=prompt,
            width=width,
            height=height,
            image_num=image_num,
            steps=steps,
            guidance_scale=guidance_scale,
            sampler_name=sampler_name,
            source_image_conditioning_scale=source_image_conditioning_scale,
            style_mode=style_mode,
            negative_prompt=negative_prompt,
            seed=seed,
            clip_skip=clip_skip,
            loras=loras,
            embeddings=embeddings
        )
        if response_image_type is not None:
            req.set_image_type(response_image_type)
        if enterprise_plan is not None:
            req.set_enterprise_plan(enterprise_plan)
        
        return InstantStyleResponse.from_dict(self._post('/v3/async/instant-style', req.to_dict()))
    
    def instant_style(self, ref_image: InputImage, model_name: str,
                      prompt: str, width: int, height: int,
                      image_num: int, steps: int, guidance_scale: float,
                      sampler_name: str, source_images: InputImage = None,
                      style_mode: int = None, source_image_conditioning_scale: float = 0.5,
                      negative_prompt: str = None, seed: str = -1, clip_skip: int = None,
                      loras: List[InstantStyleLoRA] = None, embeddings: List[InstantStyleEmbedding] = None,
                      enterprise_plan: bool = None, response_image_type: str = None,
                    ):
        res = self.async_instant_style(ref_image, model_name, prompt, width, height, image_num, steps, guidance_scale, sampler_name, source_images, style_mode, source_image_conditioning_scale, negative_prompt, seed, clip_skip, loras, embeddings, enterprise_plan, response_image_type)
        final_res = self.wait_for_task_v3(res.task_id)
        if final_res.task.status != V3TaskResponseStatus.TASK_STATUS_SUCCEED:
            logger.error(f"Task {final_res.task.task_id} failed with status {final_res.task.status}")
        else:
            final_res.download_images()
        return final_res



    def raw_img2img_v3(self, req: Img2ImgV3Request, extra: CommonV3Extra = None) -> Img2ImgV3Response:
        _req = CommonV3Request(request=req, extra=extra)

        return Img2ImgV3Response.from_dict(self._post('/v3/async/img2img', _req.to_dict()))

    def img2img_v3(self, model_name: str, input_image: str, prompt: str, image_num: int, height: int = None, width: int = None, negative_prompt: str = None, sd_vae: str = None, steps: int = None, guidance_scale: float = None, clip_skip: int = None, seed: int = None, strength: float = None, sampler_name: str = None, response_image_type: str = None, loras: List[Img2V3ImgLoRA] = None, embeddings: List[Img2ImgV3Embedding] = None, controlnet_units: List[Img2ImgV3ControlNetUnit] = None, download_images: bool = True, callback: callable = None, **kwargs) -> Img2ImgV3Response:
        input_image = input_image_to_base64(input_image)
        req = Img2ImgV3Request(
            model_name=model_name,
            image_base64=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            sd_vae=sd_vae,
            steps=steps,
            clip_skip=clip_skip,
            loras=loras,
            embeddings=embeddings,
        )
        if height is not None:
            req.height = height
        if width is not None:
            req.width = width
        if sampler_name is not None:
            req.sampler_name = sampler_name
        if image_num is not None:
            req.image_num = image_num
        if guidance_scale is not None:
            req.guidance_scale = guidance_scale
        if strength is not None:
            req.strength = strength
        if seed is not None:
            req.seed = seed
        if controlnet_units is not None:
            for unit in controlnet_units:
                unit.image_base64 = input_image_to_base64(unit.image_base64)
            req.controlnet = Img2ImgV3ControlNet(units=controlnet_units)

        #     req.set_image_type(response_image_type)
        extra = CommonV3Extra(**kwargs)
        if response_image_type is not None:
            extra.response_image_type = response_image_type

        res = self.raw_img2img_v3(req, extra)
        final_res = self.wait_for_task_v3(res.task_id, callback=callback)
        if final_res.task.status != V3TaskResponseStatus.TASK_STATUS_SUCCEED:
            logger.error(f"Task {res.task_id} failed with status {final_res.task.status}")
        else:
            if download_images:
                final_res.download_images()
        return final_res

    def raw_txt2img_v3(self, req: Txt2ImgV3Request, extra: CommonV3Extra = None) -> Txt2ImgV3Response:
        _req = CommonV3Request(request=req, extra=extra)
        return Txt2ImgV3Response.from_dict(self._post('/v3/async/txt2img', _req.to_dict()))

    def txt2img_v3(self, model_name: str, prompt: str, image_num: int, height: int = None, width: int = None, negative_prompt: str = None, sd_vae: str = None, steps: int = None, guidance_scale: float = None, clip_skip: int = None, seed: int = None, strength: float = None, sampler_name: str = None, response_image_type: str = None, loras: List[Txt2ImgV3LoRA] = None, embeddings: List[Txt2ImgV3Embedding] = None, hires_fix: Txt2ImgV3HiresFix = None, refiner: Txt2ImgV3Refiner = None, download_images: bool = True, callback: callable = None, **kwargs) -> Txt2ImgV3Response:
        req = Txt2ImgV3Request(
            model_name=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            sd_vae=sd_vae,
            clip_skip=clip_skip,
            loras=loras,
            embeddings=embeddings,
            hires_fix=hires_fix,
            refiner=refiner,
        )
        if steps is not None:
            req.steps = steps

        if height is not None:
            req.height = height
        if width is not None:
            req.width = width
        if sampler_name is not None:
            req.sampler_name = sampler_name
        if image_num is not None:
            req.image_num = image_num
        if guidance_scale is not None:
            req.guidance_scale = guidance_scale
        if strength is not None:
            req.strength = strength
        if seed is not None:
            req.seed = seed
        
        extra = CommonV3Extra(**kwargs)
        if response_image_type is not None:
            extra.response_image_type = response_image_type

        res = self.raw_txt2img_v3(req, extra)
        final_res = self.wait_for_task_v3(res.task_id, callback=callback)
        if final_res.task.status != V3TaskResponseStatus.TASK_STATUS_SUCCEED:
            logger.error(f"Task {res.task_id} failed with status {final_res.task.status}")
            raise NovitaResponseError(f"Task {res.task_id} failed with status {final_res.task.status}")
        else:
            if download_images:
                final_res.download_images()
        return final_res

    def user_info(self) -> UserInfoResponse:
        return UserInfoResponse.from_dict(self._get("/v3/user"))
    
    def query_model_v3(self, visibility:str = None, limit:str = None,\
                       query:str = None, cursor:str = None,is_inpainting:bool \
                        = False, source:str = None, is_sdxl:bool = None,\
                            types:str= None) -> MoodelsResponseV3:
        parameters = {}
        if visibility is not None:
            parameters["filter.visibility"] = visibility
        if source is not None:
            parameters["filter.source"] = source
        if types is not None:
            parameters["filter.type"] = types
        if is_sdxl is not None:
            parameters["filter.is_sdxl"] = is_sdxl
        if query is not None:
            parameters["filter.query"] = query
        if cursor is not None:
            parameters["pagination.cursor"] = f"c_{cursor}"
        parameters["filter.is_inpainting"] = is_inpainting
        if limit is not None:
            if float(limit) > 100 or float(limit) <= 0:
                limit = 1
            parameters["pagination.limit"] = limit
        res = self._get('/v3/model', params=parameters)
        return MoodelsResponseV3.from_dict(res)

    def models_v3(self, refresh=False) -> ModelListV3:
        if self._model_list_cache_v3 is None or len(self._model_list_cache_v3) == 0 or refresh:
            visibilities = ["public", "private"]
            ret = []
            for visibilitiy in visibilities: #interesting spelling :)
                offset = 0
                page_size = 100
                while True:
                    res = self._get('/v3/model', params={"pagination.cursor": f"c_{offset}", "pagination.limit": page_size, "filter.visibility": visibilitiy})
                    model_response: MoodelsResponseV3 = MoodelsResponseV3.from_dict(res)
                    for i in range(len(model_response.models)):
                        model_response.models[i].visibility = visibilitiy
                    ret.extend(model_response.models)
                    if model_response.models is None or len(model_response.models) == 0 or len(model_response.models) < page_size:
                        break
                    offset += page_size
            self._model_list_cache_v3 = ModelListV3(ret)
        return self._model_list_cache_v3

    def models(self, refresh=False) -> ModelList:
        """Get list of models

        This method retrieves a list of models available in the Novita API. If the list has already been retrieved and
        `refresh` is False, the cached list will be returned. Otherwise, a new request will be made to the API to
        retrieve the list.

        Args:
            refresh (bool, optional): If True, a new request will be made to the API to retrieve the list of models.
                If False and the list has already been retrieved, the cached list will be returned. Defaults to False.

        Returns:
            ModelList: A list of models available in the Novita API.
        """

        if (self._model_list_cache is None or len(self._model_list_cache) == 0) or refresh:
            res = self._get('/v2/models')

            # TODO: fix this
            res_controlnet = self._get(
                '/v2/models', params={'type': 'controlnet'})
            res_vae = self._get('/v2/models', params={'type': 'vae'})

            tmp = []
            tmp.extend(MoodelsResponse.from_dict(res).data.models)
            tmp.extend(MoodelsResponse.from_dict(res_controlnet).data.models)
            tmp.extend(MoodelsResponse.from_dict(res_vae).data.models)

            # In future /models maybe return all models, so we need to filter out duplicates
            tmp_set = set()
            models = []
            for m in tmp:
                if m.sd_name not in tmp_set:
                    tmp_set.add(m.sd_name)
                    models.append(m)

            self._model_list_cache = ModelList(models)

        return self._model_list_cache
