import time

from fastapi import FastAPI, Body

from utils import get_img_path, decode_base64_to_image
from scratch_remove.scratch_remove_utils import remove_scratch_using_mask
from scratch_remove.scratch_remove_utils import download_scratch_remover_model

app = FastAPI()


@app.get("/ai/api/v1/scratch-remove-server-test")
async def scratch_remove_server_test():

    return {
        "success": True,
        "message": "Server is OK."
    }


@app.post('/ai/api/v1/scratch_remove')
async def generate_mask_image(
        input_image: str = Body("", title='scratch remove input image'),
        upscale: bool = Body(False, title='input image name')
):
    start_time = time.time()

    download_scratch_remover_model()
    pil_image = decode_base64_to_image(input_image).convert("RGB")
    out_image = remove_scratch_using_mask(pil_image, upscale)

    out_images_directory_name = '/scratch_images/'

    out_image_path = get_img_path(out_images_directory_name)
    out_image.save(out_image_path)

    return {
        "success": True,
        "message": "Returned output successfully",
        "server_process_time": time.time() - start_time,
        "output_image_urls": '/media' + out_images_directory_name + out_image_path.split('/')[-1]
    }
