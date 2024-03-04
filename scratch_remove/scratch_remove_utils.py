import os
from PIL import Image
import numpy as np
import cv2
import torch

from diffusers import ControlNetModel, DEISMultistepScheduler

from scratch_remove.scratch_detection import ScratchDetection
from scratch_remove.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from scratch_remove.download_file import download_file
from scratch_remove.download_file import SCRATCH_MODEL_NAME, MODEL_PATH, scratch_model_file_download_url

device = "cuda"

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

# speed up diffusion process with faster scheduler and memory optimization
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')


def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = float(width) / float(height)
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    return image.resize((new_width, new_height), Image.BICUBIC)


def remove_scratch_using_mask(source_image: Image, upscale: bool):
    scratch_detector = ScratchDetection(source_image, input_size="scale_256", gpu=0)
    main_image, mask_image = scratch_detector.run()

    # Resize the mask to match the input image size
    mask_image = mask_image.resize(mask_image.size, Image.BICUBIC)

    # Apply dilation to make the lines bigger
    kernel = np.ones((5, 5), np.uint8)
    mask_image_np = np.array(mask_image)
    mask_image_np_dilated = cv2.dilate(mask_image_np, kernel, iterations=2)
    mask_image_dilated = Image.fromarray(mask_image_np_dilated)

    ##scratck removing
    main_image = main_image.convert("RGB")
    main_image = resize_image(main_image, 768)

    main_mask = mask_image_dilated
    main_mask = resize_image(main_mask, 768)

    image = np.array(main_image)
    low_threshold = 100
    high_threshold = 200
    canny = cv2.Canny(image, low_threshold, high_threshold)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    canny_image = Image.fromarray(canny)
    generator = torch.manual_seed(0)

    without_scratch_Image_output = pipe(
        prompt="",
        num_inference_steps=20,
        generator=generator,
        image=main_image,
        control_image=canny_image,
        controlnet_conditioning_scale=0,
        mask_image=main_mask
    ).images[0]
    return without_scratch_Image_output


def download_scratch_remover_model():
    destination = os.path.join(MODEL_PATH, SCRATCH_MODEL_NAME)
    if os.path.isfile(destination):
        return
    os.makedirs(MODEL_PATH, exist_ok=True)
    download_file(scratch_model_file_download_url, destination)
