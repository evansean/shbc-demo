import math

import numpy as np
import scipy
import skimage
import torch
import torchvision.transforms.functional as functional
from skimage.measure import label, regionprops
from torchvision.transforms import InterpolationMode

from src.resunet import SegmentationModel


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


def load_model(path, adapt_mode='test-bn'):
    print(f"Loading segmentation model... (adaptation mode='{adapt_mode}')")
    model = SegmentationModel(num_foregrounds=2)
    model.load_state_dict(torch.load(path))
    model = model.to(get_device())

    model.eval()
    if adapt_mode == 'test-bn':
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.training = True

    return model


def process_np_image(np_img, size, use_center_crop=True):
    # input: numpy image of shape H x W x C
    # output: processed numpy image of shape H x W x C and image tensor of shape 1 x C x H x W
    img_tensor = torch.from_numpy(np_img.copy()).permute(2, 0, 1).float()
    img_tensor = functional.resize(img_tensor, size)
    if use_center_crop:
        img_tensor = functional.center_crop(img_tensor, size)
    np_img = img_tensor.permute(1, 2, 0).int().numpy().astype(np.uint8)
    img_tensor = (img_tensor / 255).unsqueeze(0)

    return np_img, img_tensor


def get_prediction(seg_model, img_tensor):
    # input: image tensor of shape B x C x H x W
    # output: probability tensor of shape B x H x W
    device = get_device()
    img_tensor = img_tensor.to(device)
    height = img_tensor.shape[-2]
    width = img_tensor.shape[-1]

    disc_cup_preds = []
    with torch.no_grad():
        for zoom_level in [0.8, 0.9, 1.0]:
            zoom_height = int(zoom_level * height)
            zoom_width = int(zoom_level * width)
            zoom_tensor = functional.center_crop(img_tensor, [zoom_height, zoom_width])
            zoom_tensor = functional.resize(zoom_tensor, [height, width])
            for rotation in [0, 90, 180, 270]:
                rotated = functional.rotate(zoom_tensor, rotation, InterpolationMode.BILINEAR)
                disc_cup = seg_model.inference(rotated, height, width)
                disc_cup = functional.rotate(disc_cup, -rotation)
                disc_cup = functional.resize(disc_cup, [zoom_height, zoom_width])
                disc_cup = functional.pad(disc_cup, [(height - zoom_height) // 2, (width - zoom_width) // 2])
                disc_cup_preds.append(disc_cup)
            for flip_fn in [functional.hflip, functional.vflip]:
                flipped = flip_fn(zoom_tensor)
                disc_cup = seg_model.inference(flipped, height, width)
                disc_cup = flip_fn(disc_cup)
                disc_cup = functional.resize(disc_cup, [zoom_height, zoom_width])
                disc_cup = functional.pad(disc_cup, [(height - zoom_height) // 2, (width - zoom_width) // 2])
                disc_cup_preds.append(disc_cup)
    disc_cup_prediction = sum(disc_cup_preds) / len(disc_cup_preds)

    return disc_cup_prediction.squeeze().cpu()


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


def get_contour(np_mask):
    # input: numpy image of shape H x W
    # output: numpy image of shape H x W x 1
    boundary_coords = None
    contours = skimage.measure.find_contours(np_mask)
    for coords in contours:
        if boundary_coords is None or len(coords) > len(boundary_coords):
            boundary_coords = coords

    boundary_mask = np.zeros((np_mask.shape[0], np_mask.shape[1], 1), int)
    if boundary_coords is not None:
        rc, cc = boundary_coords[:, 0].astype(int), boundary_coords[:, 1].astype(int)
        boundary_mask[rc, cc] = 1

    filled_mask = np.zeros((np_mask.shape[0], np_mask.shape[1], 1), int)
    if boundary_coords is not None:
        rc, cc = skimage.draw.polygon(boundary_coords[:, 0], boundary_coords[:, 1], boundary_mask.shape)
        filled_mask[rc, cc] = 1

    return boundary_mask, filled_mask


def zoom(np_img, midpoint, size):
    # input: numpy image of shape H x W x C
    # output: numpy image of shape H x W x C
    start_row = max(0, midpoint[0] - size // 2)
    end_row = min(np_img.shape[0], start_row + size)
    start_col = max(0, midpoint[1] - size // 2)
    end_col = min(np_img.shape[1], start_col + size)

    return np_img[start_row:end_row, start_col:end_col, :]


def check_violation(disc_mask, cup_mask):
    disc_mask = torch.tensor(disc_mask).view(1, 1, disc_mask.shape[0], disc_mask.shape[1]).float()
    cup_mask = torch.tensor(cup_mask).view(1, 1, cup_mask.shape[0], cup_mask.shape[1]).float()
    disc_mask = ((disc_mask - cup_mask) > 0).float()
    background_mask = torch.ones_like(disc_mask) - disc_mask - cup_mask
    kernel = torch.ones(1, 1, 3, 3)
    background_neighbors = (torch.nn.functional.conv2d(background_mask, kernel, padding=1) > 0).int()
    cup_neighbors = (torch.nn.functional.conv2d(cup_mask, kernel, padding=1) > 0).int()
    critical_mask = background_neighbors * cup_mask + cup_neighbors * background_mask

    return critical_mask.sum().item() >= 1


def fill_hole(mask):
    label_image = label(mask)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        mask[label_image != idx_max + 1] = 0
    processed = scipy.ndimage.binary_fill_holes(np.asarray(mask).astype(int))

    return processed


def compute_line_length(line_mask):
    coordinates = [(r, c) for r, c in zip(*np.nonzero(line_mask))]
    length = 0
    for i in range(len(coordinates)):
        r1, c1 = coordinates[i]
        for j in range(i + 1, len(coordinates)):
            r2, c2 = coordinates[j]
            length = max(length, np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2))

    return length


def get_cdr_vector(midpoint, radius, disc_mask, cup_mask, degree_step=30):
    cdr_vector = dict()
    mid_row, mid_col = midpoint
    disc_lines = np.zeros_like(disc_mask)
    cup_lines = np.zeros_like(disc_mask)
    for degree in range(0, 180, degree_step):
        # calculate coordinate of two endpoints on the circle perimeter
        radian = math.radians(degree)
        row_offset = radius * math.sin(radian)
        col_offset = radius * math.cos(radian)
        point_1 = (int(mid_row - row_offset), int(mid_col - col_offset))
        point_2 = (int(mid_row + row_offset), int(mid_col + col_offset))

        # calculate the length of the intersected od/oc
        line = np.zeros_like(disc_mask)
        rr, cc = skimage.draw.line(*point_1, *point_2)
        while rr[0] < 0 or cc[0] < 0 or rr[0] >= disc_mask.shape[0] or cc[0] >= disc_mask.shape[1]:
            rr = rr[1:]
            cc = cc[1:]
        while rr[-1] < 0 or cc[-1] < 0 or rr[-1] >= disc_mask.shape[0] or cc[-1] >= disc_mask.shape[1]:
            rr = rr[:-1]
            cc = cc[:-1]
        line[rr, cc] = 1
        disc_line = line * disc_mask
        # disc_length = disc_line.sum()
        disc_length = compute_line_length(disc_line[:, :, 0])
        cup_line = line * cup_mask
        # cup_length = cup_line.sum()
        cup_length = compute_line_length(cup_line[:, :, 0])
        disc_lines += disc_line
        cup_lines += cup_line

        # add to cdr vector
        if disc_length == 0:
            cdr_vector[degree] = -1
        else:
            cdr_vector[degree] = round(cup_length / disc_length, 4)

    return cdr_vector, (disc_lines > 0).astype(int), (cup_lines > 0).astype(int)


def get_sector_wise_cdr_vector(midpoint, radius, disc_mask, cup_mask, degree_step=30):
    scdr_vector = dict()
    rdr_vector = dict()
    mid_row, mid_col = midpoint
    for degree in range(0, 180, degree_step):
        # calculate coordinate of two endpoints on the circle perimeter
        radian = math.radians(degree)
        row_offset = radius * math.sin(radian)
        col_offset = radius * math.cos(radian)
        point_1 = (int(mid_row - row_offset), int(mid_col - col_offset))
        point_2 = (int(mid_row + row_offset), int(mid_col + col_offset))

        # draw section lines
        line1 = np.zeros_like(disc_mask)
        rr1, cc1 = skimage.draw.line(*point_1, *midpoint)
        line2 = np.zeros_like(disc_mask)
        rr2, cc2 = skimage.draw.line(*point_2, *midpoint)

        # remove out-of-bound points
        while rr1[0] < 0 or cc1[0] < 0 or rr1[0] >= disc_mask.shape[0] or cc1[0] >= disc_mask.shape[1]:
            rr1 = rr1[1:]
            cc1 = cc1[1:]
        while rr1[-1] < 0 or cc1[-1] < 0 or rr1[-1] >= disc_mask.shape[0] or cc1[-1] >= disc_mask.shape[1]:
            rr1 = rr1[:-1]
            cc1 = cc1[:-1]
        while rr2[0] < 0 or cc2[0] < 0 or rr2[0] >= disc_mask.shape[0] or cc2[0] >= disc_mask.shape[1]:
            rr2 = rr2[1:]
            cc2 = cc2[1:]
        while rr2[-1] < 0 or cc2[-1] < 0 or rr2[-1] >= disc_mask.shape[0] or cc2[-1] >= disc_mask.shape[1]:
            rr2 = rr2[:-1]
            cc2 = cc2[:-1]
        line1[rr1, cc1] = 1
        line2[rr2, cc2] = 1

        disc_line1 = line1 * disc_mask
        cup_line1 = line1 * cup_mask
        # disc_line1_length = disc_line1.sum()
        # cup_line1_length = cup_line1.sum()
        disc_line1_length = compute_line_length(disc_line1[:, :, 0])
        cup_line1_length = compute_line_length(cup_line1[:, :, 0])
        mapped_degree1 = 180 - degree
        if mapped_degree1 < 0:
            mapped_degree1 += 360
        if disc_line1_length == 0:
            scdr_vector[mapped_degree1] = -1
        else:
            scdr_vector[mapped_degree1] = round(cup_line1_length / disc_line1_length, 4)
        disc_line2 = line2 * disc_mask
        cup_line2 = line2 * cup_mask
        # disc_line2_length = disc_line2.sum()
        # cup_line2_length = cup_line2.sum()
        disc_line2_length = compute_line_length(disc_line2[:, :, 0])
        cup_line2_length = compute_line_length(cup_line2[:, :, 0])
        mapped_degree2 = -degree
        if mapped_degree2 < 0:
            mapped_degree2 += 360
        if disc_line2_length == 0:
            scdr_vector[mapped_degree2] = -1
        else:
            scdr_vector[mapped_degree2] = round(cup_line2_length / disc_line2_length, 4)
        if disc_line1_length + disc_line2_length == 0:
            rdr_vector[mapped_degree1] = -1
            rdr_vector[mapped_degree2] = -1
        else:
            rdr_vector[mapped_degree1] = round(
                (disc_line1_length - cup_line1_length) / (disc_line1_length + disc_line2_length), 4
            )
            rdr_vector[mapped_degree2] = round(
                (disc_line2_length - cup_line2_length) / (disc_line1_length + disc_line2_length), 4)

    return scdr_vector, rdr_vector


def analyze_disc_cup(np_img, pred, apply_hflip):
    if apply_hflip:
        np_img = np.flip(np_img, axis=1)
        pred = torch.flip(pred, dims=[2])
    np_disc_mask = fill_hole((pred[0] >= 0.5).numpy())
    np_cup_mask = fill_hole((pred[1] >= 0.5).numpy())

    if np_disc_mask.sum() == 0:
        np_disc_intensities = (pred[0] * 255).numpy().astype(np.uint8)
        disc_threshold = skimage.filters.threshold_otsu(np_disc_intensities)
        np_disc_mask = fill_hole(np_disc_intensities > disc_threshold)
    if np_cup_mask.sum() == 0:
        np_cup_intensities = (pred[1] * 255).numpy().astype(np.uint8)
        cup_threshold = skimage.filters.threshold_otsu(np_cup_intensities)
        np_cup_mask = fill_hole(np_cup_intensities > cup_threshold)

    disc_contour, np_disc_mask = get_contour(np_disc_mask)
    disc_top, disc_bottom, disc_left, disc_right = get_bounding_box(disc_contour)
    midpoint = (
        disc_top + (disc_bottom - disc_top) // 2,
        disc_left + (disc_right - disc_left) // 2
    )

    cup_contour, np_cup_mask = get_contour(np_cup_mask)
    cup_top, cup_bottom, cup_left, cup_right = get_bounding_box(cup_contour)

    cup_height = cup_bottom - cup_top
    disc_height = disc_bottom - disc_top
    disc_width = disc_right - disc_left
    vcdr = round(cup_height / disc_height, 4)
    violation = check_violation(np_disc_mask, np_cup_mask)
    cup_midpoint = (
        cup_top + (cup_bottom - cup_top) // 2,
        cup_left + (cup_right - cup_left) // 2
    )
    cdr_vector, disc_lines, cup_lines = get_cdr_vector(
        cup_midpoint, max(disc_height, disc_width), np_disc_mask, np_cup_mask
    )
    scdr_vector, rdr_vector = get_sector_wise_cdr_vector(
        cup_midpoint, max(disc_height, disc_width), np_disc_mask, np_cup_mask
    )
    info = {
        "disc_height": disc_height,
        "disc_width": disc_width,
        "cup_height": cup_height,
        "cup_width": cup_right - cup_left,
        "midpoint": midpoint,
        "vcdr": vcdr,
        "cdr_vector": cdr_vector,
        "scdr_vector": scdr_vector,
        "rdr_vector": rdr_vector,
        "violation": violation
    }

    cup_contour = np.tile(cup_contour, (1, 1, 3)) * 255
    disc_contour = np.tile(disc_contour, (1, 1, 3)) * 255
    if not violation:
        disc_contour[:, :, 0] = 0
        disc_contour[:, :, 2] = 0
    else:
        disc_contour[:, :, 1] = 0
        disc_contour[:, :, 2] = 0
    segmentation_result = np.clip(np_img - cup_contour + disc_contour, 0, 255)

    disc_lines -= cup_lines
    cup_lines = np.tile(cup_lines, (1, 1, 3)) * 255
    disc_lines = np.tile(disc_lines, (1, 1, 3)) * 255
    disc_lines[:, :, 0] = 0
    disc_lines[:, :, 2] = 0
    cdr_vector_result = np.clip(np_img - cup_lines + disc_lines, 0, 255)

    return segmentation_result, cdr_vector_result, info
