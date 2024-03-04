import gc
import os
import warnings
import torch

import torch.nn.functional as F
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torchvision as tv

from detection_models import networks
from scratch_remove_utils import SCRATCH_MODEL_NAME, MODEL_PATH

warnings.filterwarnings("ignore", category=UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    elif full_size == "scale_256":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")


def process_image(scratched_image, input_size="scale_256", gpu=0):
    print("initializing the dataloader")

    # Initialize the model
    model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )

    ## load model
    checkpoint_path = os.path.join(MODEL_PATH, SCRATCH_MODEL_NAME)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print("model weights loaded")

    if gpu >= 0:
        model.to(gpu)
    else:
        model.cpu()
    model.eval()

    transformed_image_PIL = data_transforms(scratched_image, input_size)
    scratch_image = transformed_image_PIL.convert("L")
    scratch_image = tv.transforms.ToTensor()(scratch_image)
    scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
    scratch_image = torch.unsqueeze(scratch_image, 0)
    _, _, ow, oh = scratch_image.shape
    scratch_image_scale = scale_tensor(scratch_image)

    if gpu >= 0:
        scratch_image_scale = scratch_image_scale.to(gpu)
    else:
        scratch_image_scale = scratch_image_scale.cpu()
    with torch.no_grad():
        P = torch.sigmoid(model(scratch_image_scale))

    P = P.data.cpu()
    P = F.interpolate(P, [ow, oh], mode="nearest")
    gc.collect()
    torch.cuda.empty_cache()
    transform = transforms.ToPILImage()
    pil_mask = transform(P.squeeze())
    return transformed_image_PIL, pil_mask


class ScratchDetection:
    def __init__(self, scratched_image, input_size="scale_256", gpu=0):
        self.scratched_image = scratched_image
        self.input_size = input_size
        self.gpu = gpu

    def run(self):
        pil_image, pil_mask = process_image(self.scratched_image, self.input_size, self.gpu)
        return pil_image, pil_mask
