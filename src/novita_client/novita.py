#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging

from time import sleep

from .version import __version__

from .exceptions import *
from .proto import *

import requests
from . import settings
from .utils import input_image_to_base64


logger = logging.getLogger(__name__)


class NovitaClient:
    """NovitaClient is the main entry point for interacting with the Novita API."""

    def __init__(self, api_key):
        self.base_url = "https://api.novita.ai"
        self.api_key = api_key
        self.session = requests.Session()

        if not self.api_key:
            raise ValueError("NOVITA_API_KEY environment variable not set")

        # eg: {"all": [proto.ModelInfo], "checkpoint": [proto.ModelInfo], "lora": [proto.ModelInfo]}
        self._model_list_cache = None
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

        logger.debug(f"[GET] response: {response.content}")
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

        logger.debug(f"[POST] data: {data}")

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

    def txt2img(self, request: Txt2ImgRequest) -> Txt2ImgResponse:
        """Asynchronously generate images from request

        Args:
            request (Txt2ImgRequest): The request object containing the text and image generation parameters.

        Returns:
            Txt2ImgResponse: The response object containing the task ID and status URL.
        """
        response = self._post('/v2/txt2img', request.to_dict())

        return Txt2ImgResponse.from_dict(response)

    def progress(self, task_id: str) -> ProgressResponse:
        """Get the progress of a task.

        Args:
            task_id (str): The ID of the task to get the progress for.

        Returns:
            ProgressResponse: The response object containing the progress information for the task.
        """
        response = self._get('/v2/progress', {
            'task_id': task_id,
        })

        return ProgressResponse.from_dict(response)

    def async_task_result(self, task_id: str) -> V3TaskResponse:
        response = self._get('/v3/async/task-result', {
            'task_id': task_id,
        })
        return V3TaskResponse.from_dict(response)

    def img2img(self, request: Img2ImgRequest) -> Img2ImgResponse:
        """Asynchronously generate images from request

        Args:
            request (Img2ImgRequest): The request object containing the image and image generation parameters.

        Returns:
            Img2ImgResponse: The response object containing the task ID and status URL.
        """
        response = self._post('/v2/img2img', request.to_dict())

        return Img2ImgResponse.from_dict(response)

    def wait_for_task(self, task_id, wait_for: int = 300, callback: callable = None) -> ProgressResponse:
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

            progress = self.progress(task_id)

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
                return progress

            sleep(settings.DEFAULT_POLL_INTERVAL)
            i += 1

    def sync_txt2img(self, request: Txt2ImgRequest, download_images=True, callback: callable = None) -> ProgressResponse:
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
        response = self.txt2img(request)

        if response.data is None:
            raise NovitaResponseError(f"Text to Image generation failed with response {response.msg}, code: {response.code}")

        res = self.wait_for_task(response.data.task_id, callback=callback)
        if download_images:
            res.download_images()
        return res

    def sync_img2img(self, request: Img2ImgRequest, download_images=True, callback: callable = None) -> ProgressResponse:
        """Synchronously generate images from request, optionally download images

        Args:
            request (Img2ImgRequest): The request object containing the input image and other parameters.
            download_images (bool, optional): Whether to download the generated images. Defaults to True.

        Returns:
            ProgressResponse: The response object containing the task status and generated images.
        """
        response = self.img2img(request)

        if response.data is None:
            raise NovitaResponseError(f"Image to Image generation failed with response {response.msg}, code: {response.code}")

        res = self.wait_for_task(response.data.task_id, callback=callback)
        if download_images:
            res.download_images()
        return res

    def sync_upscale(self, request: UpscaleRequest, download_images=True, callback: callable = None) -> ProgressResponse:
        """Syncronously upscale image from request, optionally download images

        Args:
            request (UpscaleRequest): _description_
            download_images (bool, optional): _description_. Defaults to True.

        Returns:
            ProgressResponse: _description_
        """
        response = self.upscale(request)

        if response.data is None:
            raise NovitaResponseError(f"Upscale failed with response {response.msg}, code: {response.code}")

        res = self.wait_for_task(response.data.task_id, callback=callback)
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

    def cleanup(self, image: InputImage, mask: InputImage, response_image_type=None) -> CleanupResponse:
        image_b64 = input_image_to_base64(image)
        mask_b64 = input_image_to_base64(mask)
        request = CleanupRequest(image_file=image_b64, mask_file=mask_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)

        return CleanupResponse.from_dict(self._post('/v3/cleanup', request.to_dict()))

    def outpainting(self, image: InputImage, width=None, height=None, center_x=None, center_y=None, response_image_type=None) -> OutpaintingResponse:
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

        request.set_image_type(self._default_response_image_type)
        return OutpaintingResponse.from_dict(self._post('/v3/outpainting', request.to_dict()))

    def remove_background(self, image: InputImage, response_image_type=None) -> RemoveBackgroundResponse:
        image_b64 = input_image_to_base64(image)
        request = RemoveBackgroundRequest(image_file=image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)

        return RemoveBackgroundResponse.from_dict(self._post('/v3/remove-background', request.to_dict()))

    def remove_text(self, image: InputImage, response_image_type=None) -> RemoveTextResponse:
        image_b64 = input_image_to_base64(image)
        request = RemoveTextRequest(image_file=image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)

        return RemoveTextResponse.from_dict(self._post('/v3/remove-text', request.to_dict()))

    def reimagine(self, image: InputImage, response_image_type=None) -> ReimagineResponse:
        image_b64 = input_image_to_base64(image)
        request = ReimagineRequest(image_file=image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)

        return ReimagineResponse.from_dict(self._post('/v3/reimagine', request.to_dict()))

    def doodle(self, image: InputImage, prompt: str, response_image_type=None) -> DoodleResponse:
        image_b64 = input_image_to_base64(image)
        request = DoodleRequest(image_file=image_b64, prompt=prompt)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)

        return DoodleResponse.from_dict(self._post('/v3/doodle', request.to_dict()))

    def mixpose(self, image: InputImage, pose_image: InputImage, response_image_type=None) -> MixPoseResponse:
        image_b64 = input_image_to_base64(image)
        pose_image_b64 = input_image_to_base64(pose_image)
        request = MixPoseRequest(image_file=image_b64, pose_image_file=pose_image_b64)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)

        return MixPoseResponse.from_dict(self._post('/v3/mix-pose', request.to_dict()))

    def replace_background(self, image: InputImage, prompt: str, response_image_type=None) -> ReplaceBackgroundResponse:
        image_b64 = input_image_to_base64(image)
        request = ReplaceBackgroundRequest(image_file=image_b64, prompt=prompt)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        return ReplaceBackgroundResponse.from_dict(self._post('/v3/replace-background', request.to_dict()))

    def replace_sky(self, image: InputImage, sky: str, response_image_type=None) -> ReplaceSkyResponse:
        image_b64 = input_image_to_base64(image)
        request = ReplaceSkyRequest(image_file=image_b64, sky=sky)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        return ReplaceSkyResponse.from_dict(self._post('/v3/replace-sky', request.to_dict()))

    def replace_object(self, image: InputImage, object_prompt: str, prompt: str, negative_prompt=None, response_image_type=None) -> ReplaceObjectResponse:
        res: V3AsyncSubmitResponse = self.async_replace_object(image, object_prompt, prompt, negative_prompt, response_image_type)
        final_res = self.wait_for_task_v3(res.task_id)
        final_res.download_images()
        return ReplaceObjectResponse(
            image_file=final_res.images_encoded[0],
            image_type=final_res.images[0].image_type,
        )

    def async_replace_object(self, image: InputImage, object_prompt: str, prompt: str, negative_prompt=None, response_image_type=None) -> ReplaceObjectResponse:
        image_b64 = input_image_to_base64(image)
        request = ReplaceObjectRequest(image_file=image_b64, object_prompt=object_prompt, prompt=prompt, negative_prompt=negative_prompt)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        return V3AsyncSubmitResponse.from_dict(self._post('/v3/async/replace-object', request.to_dict()))

    def restore_face(self, image: InputImage, fidelity=None, response_image_type=None) -> RestoreFaceResponse:
        image_b64 = input_image_to_base64(image)
        request = RestoreFaceRequest(image_file=image_b64)
        if fidelity is not None:
            request.fidelity = fidelity
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        return RestoreFaceResponse.from_dict(self._post('/v3/restore-face', request.to_dict()))

    def create_tile(self, prompt: str, negative_prompt=None, width=None, height=None, response_image_type=None) -> CreateTileResponse:
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
        return CreateTileResponse.from_dict(self._post('/v3/create-tile', request.to_dict()))

    def merge_face(self, image: InputImage, face_image: InputImage, response_image_type=None) -> MergeFaceResponse:
        input_image = input_image_to_base64(image)
        face_image = input_image_to_base64(face_image)
        request = MergeFaceRequest(image_file=input_image, face_image_file=face_image)
        if response_image_type is None:
            request.set_image_type(self._default_response_image_type)
        else:
            request.set_image_type(response_image_type)
        return MergeFaceResponse.from_dict(self._post('/v3/merge-face', request.to_dict()))

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
            res = self._get('/models')

            # TODO: fix this
            res_controlnet = self._get(
                '/models', params={'type': 'controlnet'})
            res_vae = self._get('/models', params={'type': 'vae'})

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
