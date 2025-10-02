# inference_utils.py
import os
import torch
import numpy as np
from PIL import Image
from src import det_utils, seg_utils

# DET_CHECKPOINT_PATH = "models/det.bin"
# SEG_CHECKPOINT_PATH = "models/seg.bin"
DET_SIZE = 800
SEG_SIZE = 256

# # preload models once
# det_model = det_utils.load_model(DET_CHECKPOINT_PATH, adapt_mode=None)
# seg_model = seg_utils.load_model(SEG_CHECKPOINT_PATH, adapt_mode=None)

def run_inference_on_image(image_path, det_model, seg_model, output_folder="output_single", use_padding=False):
    os.makedirs(output_folder, exist_ok=True)
    disc_cup_folder = os.path.join(output_folder, "disc_cup_segmentation")
    os.makedirs(disc_cup_folder, exist_ok=True)

    # load image
    image = Image.open(image_path).convert("RGB")
    np_img = np.array(image).astype(np.uint8)
    img_tensor = torch.from_numpy(np_img.copy()).permute(2, 0, 1).float()

    # detection
    np_img, img_tensor = det_utils.process_np_image(np_img, DET_SIZE, use_padding=use_padding)
    disc_pred = det_utils.get_prediction(det_model, img_tensor)
    bbox_result, np_img_roi = det_utils.analyze_disc(np_img, disc_pred)

    # segmentation
    np_img_roi, img_tensor = seg_utils.process_np_image(np_img_roi, SEG_SIZE)
    disc_cup_pred = seg_utils.get_prediction(seg_model, img_tensor)
    disc_cup_seg_result, cdr_vector_result, disc_cup_info = seg_utils.analyze_disc_cup(
        np_img_roi,
        disc_cup_pred,
        apply_hflip="LE" in os.path.basename(image_path)
    )

    # save segmentation result
    disc_cup_seg_result = seg_utils.zoom(
        disc_cup_seg_result, disc_cup_info["midpoint"], size=int(3.5 * disc_cup_info["disc_width"])
    )
    seg_output_path = os.path.join(disc_cup_folder, os.path.basename(image_path))
    Image.fromarray(disc_cup_seg_result.astype(np.uint8)).save(seg_output_path)

    return seg_output_path, float(disc_cup_info["vcdr"])
