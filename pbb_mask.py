import torch
import numpy as np

def add_mask(images):

    data = np.ones((256, 256))

    x, y = np.meshgrid(np.arange(256), np.arange(256))

    center_x = 0  
    center_y = 0  
    circle_radius = 256

    circle_distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    inside_circle = circle_distance <= circle_radius

    data[circle_distance >= circle_radius] = 0

    '''
    5:45-50                              
    start_angle = np.pi / 4  # start  
    end_angle = np.pi / 3.6  # end      
    10:50-60
    start_angle = np.pi / 3.6  # start
    end_angle = np.pi / 3  # end 
    15:45-60
    start_angle = np.pi / 4  # start 
    end_angle = np.pi / 3  # end      
    20:40-60
    start_angle = np.pi / 4.5  # start    
    end_angle = np.pi / 3  # end 
    '''

    start = np.random.uniform(3.5, 4.6)
    end = np.random.uniform(2.9, 3.5)
    start_angle = np.pi / start  
    end_angle = np.pi / end  
    
    pixel_angle = np.arctan2(y - center_y, x - center_x)

    inside_sector = inside_circle & (pixel_angle >= start_angle) & (pixel_angle <= end_angle)
    data[inside_sector] = 0

    mask_images = images * data

    return mask_images, data


def make_truth(images):

    data = np.ones((256, 256))

    x, y = np.meshgrid(np.arange(256), np.arange(256))

    center_x = 0  
    center_y = 0  
    circle_radius = 256

    circle_distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    inside_circle = circle_distance <= circle_radius

    data[circle_distance >= circle_radius] = 0

    new_truth = images * data

    return new_truth
  

