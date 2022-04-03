#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import torch
import torchvision
import clip
import torch.nn.functional as F
import sys

from notebook_utils import TextEncoder, load_model, get_generated_images_by_texts

model = "cc3m_cc12m_yfcc"
model_dir = f"weights/{model}"
clip_path = 'weights/clip'
vqvae_path = f'{model_dir}/stage1/model.pt'
# the checkpoint of CLIP will be downloaded at the first time.
print("Loading CLIP")
sys.stdout.flush()
model_clip, preprocess_clip = clip.load("ViT-B/32", device='cuda', download_root=clip_path)
model_clip = model_clip.cuda().eval()
# load stage 1 model: RQ-VAE
print("Loading stage 1")
sys.stdout.flush()
model_vqvae, _ = load_model(vqvae_path)
# load stage 2 model: RQ-Transformer
print("Loading stage 2")
sys.stdout.flush()
model_path = f'{model_dir}/stage2/model.pt'
model_ar, config = load_model(model_path, ema=False)
# move models from cpu to gpu
print("Moving to GPU")
sys.stdout.flush()
model_ar = model_ar.cuda().eval()
model_vqvae = model_vqvae.cuda().eval()
# prepare text encoder to tokenize natual languages
print("Loading TextEncoder")
sys.stdout.flush()
text_encoder = TextEncoder(tokenizer_name=config.dataset.txt_tok_name, context_length=config.dataset.context_length)

#text_prompts = 'a fat baby with a knife'
#text_prompts = sys.argv[1]
num_samples = 128
temperature = 1.0
top_k = 1024
top_p = 0.95

while True:
    text_prompts = input("> ")
    print("Generating")
    sys.stdout.flush()
    pixels = get_generated_images_by_texts(
        model_ar,
        model_vqvae,
        text_encoder,
        model_clip,
        preprocess_clip,
        text_prompts,
        num_samples,
        temperature,
        top_k,
        top_p,
    )

    num_visualize_samples = 16
    images = [pixel.cpu().numpy() * 0.5 + 0.5 for pixel in pixels]
    images = torch.from_numpy(np.array(images[:num_visualize_samples]))
    images = torch.clamp(images, 0, 1)
    grid = torchvision.utils.make_grid(images, nrow=4)

    img = Image.fromarray(np.uint8(grid.numpy().transpose([1,2,0])*255))
    img.save("gen.png")
