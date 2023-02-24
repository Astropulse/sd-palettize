import modules.scripts as scripts
import hitherdither
import gradio as gr

import cv2
import numpy as np
from PIL import Image

from modules import images
from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state

from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional

script_dir = scripts.basedir()

# Runs cv2 k_means quantization on the provided image with "k" color indexes
def palettize(img,k,d):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).convert("RGB")
        
        if d > 0:
            if k <= 64:

                img_indexed = image.quantize(colors=k, method=1, kmeans=k, dither=0).convert('RGB')

                palette = []
                for i in img_indexed.convert("RGB").getcolors(16777216): 
                    palette.append(i[1])
                palette = hitherdither.palette.Palette(palette)
                img_indexed = hitherdither.ordered.yliluoma.yliluomas_1_ordered_dithering(image, palette, order=2**d).convert('RGB')
            else:
                img_indexed = image.quantize(colors=k, method=1, kmeans=k, dither=0).convert('RGB')
        else:
            img_indexed = image.quantize(colors=k, method=1, kmeans=k, dither=0).convert('RGB')

        result = cv2.cvtColor(np.asarray(img_indexed), cv2.COLOR_RGB2BGR)
        return result

class Script(scripts.Script):
    def title(self):
        return "Palettize"
    def show(self, is_img2img):
        return True
    def ui(self, is_img2img):
        clusters = gr.Slider(minimum=2, maximum=128, step=1, label='Colors in palette', value=24)
        downscale = gr.Checkbox(label='Downscale before processing', value=True)
        with gr.Row():
            scale = gr.Slider(minimum=2, maximum=32, step=1, label='Downscale factor', value=8)
            dither = gr.Slider(minimum=0, maximum=3, step=1, label='Dithering', value=0)

        return [downscale, scale, clusters, dither]

    def run(self, p, downscale, scale, clusters, dither):
        
        if dither > 0:
            if clusters <= 64:
                print(f'Palettizing output to {clusters} colors with order {2**dither} dithering...')
            else:
                print('Palette too large, max colors for dithering is 64.')
                print(f'Palettizing output to {clusters} colors...')
        else:
            print(f'Palettizing output to {clusters} colors...')

        processed = process_images(p)

        generations = p.batch_size*p.n_iter

        for i in range(generations + int(generations > 1)):
            # Converts image from "Image" type to numpy array for cv2
            img = np.array(processed.images[i]).astype(np.uint8)

            if downscale:
                img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)), interpolation = cv2.INTER_LINEAR)

            img = palettize(img, clusters, dither)

            if downscale:
                img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation = cv2.INTER_NEAREST)

            processed.images[i] = Image.fromarray(img)
            images.save_image(processed.images[i], p.outpath_samples, "palettized", processed.seed + i, processed.prompt, opts.samples_format, info=processed.info, p=p)

            if generations > 1:
                grid = images.image_grid(processed.images[1:generations+1], p.batch_size)
                processed.images[0] = grid
            
            if opts.grid_save:
                images.save_image(processed.images[0], p.outpath_grids, "palettized", prompt=p.prompt, seed=processed.seed, grid=True, p=p)

        return processed