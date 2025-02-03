import numpy as np
import matplotlib.pyplot as plt


def round_mask():
    mask = np.ones((256, 256))
    y, x = np.ogrid[0:256, 0:256]
    round_drop_center_1 = (128, 128)
    round_drop_radius_1 = 50

    round_drop_mask_1 = (x - round_drop_center_1[0])**2 + (y - round_drop_center_1[1])**2 <= round_drop_radius_1**2
    
    round_mask = np.zeros_like(mask)
    round_mask[round_drop_mask_1] = 1

    return round_mask


def create_fan_mask(width, height, center, radius, start_angle, end_angle):
    y, x = np.ogrid[0:height, 0:width]
    angle = np.arctan2(y - center[1], x - center[0])
    angle = np.degrees(angle) % 360

    mask = np.ones((height, width))
    mask[(angle >= start_angle) & (angle <= end_angle) & (np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius)] = 0
    return mask

def fanshape_mask():
    width, height = 256, 256
    mask = np.ones((256, 256))
    fanshape_mask = np.ones_like(mask)

    fan_center = (0, 256)
    fan_radius = 200
    fan_start_angle = 270
    fan_end_angle = 360
    fan_mask = create_fan_mask(width, height, fan_center, fan_radius, fan_start_angle, fan_end_angle)

    fanshape_mask[~fan_mask.astype(bool)] = 0
    return fanshape_mask


def block_x_mask():
   
    mask = np.ones((256, 256))
    block_x_mask = np.ones_like(mask)
    block_x_mask[:, 64:192] = 0

    return block_x_mask


                                               
