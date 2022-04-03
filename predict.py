import tempfile
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import torch
import torchvision
import clip
import sys
from typing import Iterator
from cog import BasePredictor, Input, Path

from notebook_utils import TextEncoder, load_model, get_generated_images_by_texts

def log(msg):
    print(msg)
    sys.stdout.flush()

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda:0'
        model_name = "cc3m_cc12m_yfcc"
        model_dir = f"weights/{model_name}"
        clip_path = 'weights/clip'
        vqvae_path = f'{model_dir}/stage1/model.pt'
        # the checkpoint of CLIP will be downloaded at the first time.
        log("Loading CLIP")
        (self.model_clip, self.preprocess_clip) = clip.load("ViT-B/32", device='cuda', download_root=clip_path)
        self.model_clip = self.model_clip.cuda().eval()
        # load stage 1 model: RQ-VAE
        log("Loading stage 1")
        self.model_vqvae, _ = load_model(vqvae_path)
        # load stage 2 model: RQ-Transformer
        log("Loading stage 2")
        model_path = f'{model_dir}/stage2/model.pt'
        self.model_ar, self.config = load_model(model_path, ema=False)
        # move models from cpu to gpu
        log("Moving to GPU")
        self.model_ar = self.model_ar.cuda().eval()
        self.model_vqvae = self.model_vqvae.cuda().eval()
        # prepare text encoder to tokenize natual languages
        log("Loading TextEncoder")
        self.text_encoder = TextEncoder(tokenizer_name=config.dataset.txt_tok_name, context_length=config.dataset.context_length)

    def predict(self, 
        prompt: str=Input(description="Prompt for generating image"),
        num_samples: int=Input(default=128, description="Number of generated images"),
        temperature: float=Input(default=1.0, description="Temperature"),
        top_p: float=Input(default=0.95, description="top_p"),
        top_k: int=Input(default=1024, description="top_k"),
        n_keep: int=Input(default=16, description="n_keep"),
    ) -> Iterator[Path]:
        log(f"Generating images for {prompt}")
        pixels = get_generated_images_by_texts(
            self.model_ar,
            self.model_vqvae,
            self.text_encoder,
            self.model_clip,
            self.preprocess_clip,
            prompt,
            num_samples,
            temperature,
            top_k,
            top_p,
        )

        images = [pixel.cpu().numpy() * 0.5 + 0.5 for pixel in pixels]
        images = torch.from_numpy(np.array(images[:n_keep]))
        images = torch.clamp(images, 0, 1)
        to_img = lambda ary: Image.fromarray(np.uint8(ary.numpy().transpose([1,2,0])*255))
        #grid = torchvision.utils.make_grid(images, nrow=4)
        images = [to_img(img) for img in images]
        base_fn = prompt.lower().replace(' ', '_')[:50]
        outdir = Path(tempfile.mkdtemp())
        for (idx, image) in enumerate(images):
            img_fn = f"{base_fn}-{idx:02d}.png"
            img_fn = outdir + img_fn
            image.save(str(img_fn))
            yield img_fn
