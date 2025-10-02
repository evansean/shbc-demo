import numpy as np
import skimage
import torch
import torchvision.transforms.functional as functional
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode

from src.resunet import SegmentationModel


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


def load_model(path, adapt_mode='test-bn'):
    print(f"Loading detection model... (adaptation mode='{adapt_mode}')")
    model = SegmentationModel(num_foregrounds=1)
    model.load_state_dict(torch.load(path))
    model = model.to(get_device())

    model.eval()
    if adapt_mode == 'test-bn':
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.training = True

    return model


def process_np_image(np_img, size, use_center_crop=True, use_padding=False):
    # input: numpy image of shape H x W x C
    # output: processed numpy image of shape H x W x C and image tensor of shape 1 x C x H x W
    img_tensor = torch.from_numpy(np_img.copy()).permute(2, 0, 1).float()
    if use_padding:
        pad_horizontal = max(img_tensor.shape[-2], img_tensor.shape[-1]) - img_tensor.shape[-1]
        pad_vertical = max(img_tensor.shape[-2], img_tensor.shape[-1]) - img_tensor.shape[-2]
        img_tensor = torch.nn.functional.pad(
            img_tensor,
            (pad_horizontal // 2, pad_horizontal // 2, pad_vertical // 2, pad_vertical // 2)
        )
    img_tensor = functional.resize(img_tensor, size)
    if use_center_crop:
        img_tensor = functional.center_crop(img_tensor, size)
    np_img = img_tensor.permute(1, 2, 0).int().numpy().astype(np.uint8)
    img_tensor = (img_tensor / 255).unsqueeze(0)

    # # scrub text distractor in smartphone images
    # np_img[70:150, 680:770, :] = 0
    # img_tensor[:, :, 70:150, 680:770] = 0

    return np_img, img_tensor


def get_prediction(det_model, img_tensor):
    # input: image tensor of shape B x C x H x W
    # output: probability tensor of shape B x H x W
    device = get_device()
    img_tensor = img_tensor.to(device)
    height = img_tensor.shape[-2]
    width = img_tensor.shape[-1]

    disc_preds = []
    with torch.no_grad():
        for zoom_level in [0.8, 0.9, 1.0]:
            zoom_height = int(zoom_level * height)
            zoom_width = int(zoom_level * width)
            zoom_tensor = functional.center_crop(img_tensor, [zoom_height, zoom_width])
            zoom_tensor = functional.resize(zoom_tensor, [height, width])
            for rotation in [0, 90, 180, 270]:
                rotated = functional.rotate(zoom_tensor, rotation, InterpolationMode.BILINEAR)
                disc = det_model.inference(rotated, height, width)
                disc = functional.rotate(disc, -rotation)
                disc = functional.resize(disc, [zoom_height, zoom_width])
                disc = functional.pad(disc, [(height - zoom_height) // 2, (width - zoom_width) // 2])
                disc_preds.append(disc)
            for flip_fn in [functional.hflip, functional.vflip]:
                flipped = flip_fn(zoom_tensor)
                disc = det_model.inference(flipped, height, width)
                disc = flip_fn(disc)
                disc = functional.resize(disc, [zoom_height, zoom_width])
                disc = functional.pad(disc, [(height - zoom_height) // 2, (width - zoom_width) // 2])
                disc_preds.append(disc)
    disc_prediction = sum(disc_preds) / len(disc_preds)

    return disc_prediction.squeeze().cpu()


def get_bounding_box(np_img):
    # input: numpy image of shape H x W x C
    # output: four integer coordinates of the bounding box
    if len(np_img.shape) == 3:
        np_img = np_img.sum(2)

    top = 0
    bottom = np_img.shape[0]
    col_sums = np_img.sum(1)
    for i, s in enumerate(col_sums):
        if s != 0:
            top = i
            break
    for i, s in enumerate(col_sums[::-1]):
        if s != 0:
            bottom = bottom - i
            break

    left = 0
    right = np_img.shape[1]
    row_sums = np_img.sum(0)
    for i, s in enumerate(row_sums):
        if s != 0:
            left = i
            break
    for i, s in enumerate(row_sums[::-1]):
        if s != 0:
            right = right - i
            break

    return top, bottom, left, right


def compute_score(object_mask):
    contours = skimage.measure.find_contours(object_mask)
    perimeter = max([len(coords) for coords in contours])
    area = object_mask.sum()
    roundness = (4 * np.pi * area) / (perimeter ** 2)

    return roundness * np.sqrt(area)


def get_disc_blob(np_mask):
    # input: numpy image of shape H x W
    # output: numpy image of shape H x W x 1
    blobs = skimage.measure.label(np_mask)
    blob_mask = np.zeros((np_mask.shape[0], np_mask.shape[1], 1), int)
    score = 0
    for blob_id in range(1, blobs.max() + 1):
        candidate_blob = blobs == blob_id
        candidate_score = compute_score(candidate_blob)
        if candidate_score > score:
            blob_mask = candidate_blob
            score = candidate_score

    return blob_mask


def analyze_disc(np_img, pred, return_box_coord=False):
    binary_mask = (pred >= 0.5).numpy()
    if binary_mask.sum() == 0:
        intensities = (pred * 255).numpy().astype(np.uint8)
        threshold = skimage.filters.threshold_otsu(intensities)
        binary_mask = intensities > threshold

    disc_blob = get_disc_blob(binary_mask)
    disc_top, disc_bottom, disc_left, disc_right = get_bounding_box(disc_blob)
    midpoint = (
        disc_top + (disc_bottom - disc_top) // 2,
        disc_left + (disc_right - disc_left) // 2
    )
    # bbox_offset = int(1.3 * (disc_bottom - disc_top))
    bbox_offset = max(int(1.1 * (disc_bottom - disc_top)), 64)
    left = max(0, midpoint[1] - bbox_offset)
    upper = max(0, midpoint[0] - bbox_offset)
    right = min(np_img.shape[1], midpoint[1] + bbox_offset)
    lower = min(np_img.shape[0], midpoint[0] + bbox_offset)
    bbox = (left, upper, right, lower)
    bbox_result = Image.fromarray(np_img)
    draw = ImageDraw.Draw(bbox_result)
    draw.rectangle(bbox, outline=(0, 255, 0))

    # determine amount of padding
    left_pad = -1 * min(0, midpoint[1] - bbox_offset)
    upper_pad = -1 * min(0, midpoint[0] - bbox_offset)
    right_pad = max(np_img.shape[1], midpoint[1] + bbox_offset) - np_img.shape[1]
    lower_pad = max(np_img.shape[0], midpoint[0] + bbox_offset) - np_img.shape[0]
    paddings = ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0))

    if return_box_coord:
        return (upper, lower, left, right), paddings
    else:
        roi = np_img[upper:lower, left:right, :]
        roi = np.pad(roi, paddings)

        return bbox_result, roi
