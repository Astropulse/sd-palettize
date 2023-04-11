import modules.scripts as scripts
import hitherdither
import gradio as gr

import cv2
import numpy as np
from PIL import Image

from modules import images
from modules.processing import process_images
from modules.shared import opts

script_dir = scripts.basedir()

def adjust_gamma(image, gamma=1.0):
    # Create a lookup table for the gamma function
    gamma_map = [255 * ((i / 255.0) ** (1.0 / gamma)) for i in range(256)]
    gamma_table = bytes([(int(x / 255.0 * 65535.0) >> 8) for x in gamma_map] * 3)

    # Apply the gamma correction using the lookup table
    return image.point(gamma_table)

# Runs cv2 k_means quantization on the provided image with "k" color indexes
def palettize(input, colors, palImg, dithering, strength):
    img = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).convert("RGB")

    dithering += 1

    if palImg is not None:
        palImg = cv2.cvtColor(palImg, cv2.COLOR_BGR2RGB)
        palImg = Image.fromarray(palImg).convert("RGB")
        numColors = len(palImg.getcolors(16777216))
    else:
        numColors = colors

    palette = []

    threshold = (16*strength)/4
    
    if palImg is not None:

        numColors = len(palImg.getcolors(16777216))

        if strength > 0:
            img = adjust_gamma(img, 1.0-(0.02*strength))
            for i in palImg.getcolors(16777216): 
                palette.append(i[1])
            palette = hitherdither.palette.Palette(palette)
            img_indexed = hitherdither.ordered.bayer.bayer_dithering(img, palette, [threshold, threshold, threshold], order=2**dithering).convert('RGB')
        else:
            for i in palImg.getcolors(16777216):
                palette.append(i[1][0])
                palette.append(i[1][1])
                palette.append(i[1][2])
            palImg = Image.new('P', (256, 1))
            palImg.putpalette(palette)
            img_indexed = img.quantize(method=1, kmeans=numColors, palette=palImg, dither=0).convert('RGB')
    elif colors > 0:

        if strength > 0:
            img_indexed = img.quantize(colors=colors, method=1, kmeans=colors, dither=0).convert('RGB')
            img = adjust_gamma(img, 1.0-(0.03*strength))
            for i in img_indexed.convert("RGB").getcolors(16777216): 
                palette.append(i[1])
            palette = hitherdither.palette.Palette(palette)
            img_indexed = hitherdither.ordered.bayer.bayer_dithering(img, palette, [threshold, threshold, threshold], order=2**dithering).convert('RGB')

        else:
            img_indexed = img.quantize(colors=colors, method=1, kmeans=colors, dither=0).convert('RGB')

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
        with gr.Row():
            dither = gr.Dropdown(choices=["Bayer 2x2", "Bayer 4x4", "Bayer 8x8"], label="Matrix Size", value="Bayer 8x8", type="index")
            ditherStrength = gr.Slider(minimum=0, maximum=10, step=1, label='Dithering Strength', value=0)
        with gr.Row():
            palette = gr.Image(label="Palette image")

        return [downscale, scale, palette, clusters, dither, ditherStrength]

    def run(self, p, downscale, scale, palette, clusters, dither, ditherStrength):
        
        if ditherStrength > 0:
            if clusters <= 64:
                print(f'Palettizing output to {clusters} colors with order {2**(dither+1)} dithering...')
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

            img = palettize(img, clusters, palette, dither, ditherStrength)

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