"""
utils.py - Utility Functions

This module contains a collection of utility functions for common tasks in the project.
These functions include file handling, data processing, and configuration loading.

Functions:
- `load_config`: Load configuration settings from a YAML file.
- `reate_grid_with_color_mapping`: Create a grid pattern on an NDVI image and optionally apply color mapping.
- `extract_vineyard_health_data`: Extracts vineyard health-related data from an image.


Usage:
    import utils

    # Example usage of load_config function:
    config = util.load_config("config.yaml")
"""

import cv2
import numpy as np
import yaml
from tqdm import tqdm

def load_config(config_file):
    """Load configuration from a YAML file.
    Args:
        config_file (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """

    # Open the YAML configuration file in read mode
    with open(config_file, "r") as f:
        # Parse the YAML content and load it into a dictionary
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("_____Current configuration_____")
    
    # Iterate through the configuration dictionary and print key-value pairs
    for key, value in config.items():
        print(key + ": " + str(value))
    
    # Return the loaded configuration as a dictionary
    return config


def create_grid_with_color_mapping(ndvi_image, dim_x, dim_y):
    """
    Create a grid pattern on an NDVI image and optionally apply color mapping.

    Args:
        ndvi_image (numpy.ndarray): NDVI image as a NumPy array.
        dim_x (int): Width of grid cells.
        dim_y (int): Height of grid cells.
        COLOR (bool, optional): Whether to apply color mapping. Default is False.

    Returns:
        numpy.ndarray: Processed image with the grid pattern and optional color mapping.
    """
    # Make a copy of the NDVI image as a NumPy array
    ndvi_img_array = np.array(ndvi_image)

    # Get the dimensions of the NDVI image
    height, width = ndvi_img_array.shape

    # Create an empty image for color mapping
    color_mapped_ndvi = np.zeros((height, width), dtype=np.uint8)

    # Create an empty image for the visible grid
    image_with_visible_grid = np.copy(color_mapped_ndvi)

    # Iterate through each grid square of the image
    for y in range(0, height, dim_y):
        for x in range(0, width, dim_x):
            square = ndvi_img_array[y:y+dim_y, x:x+dim_x]
            mean_value = np.mean(square)
            
            # Map the mean value to a color intensity
            mean_value = int(mean_value * 255 * 20)
            color_mapped_ndvi[y:y+dim_y, x:x+dim_x] = mean_value

            image_without_grid = np.copy(color_mapped_ndvi)
            
            # Draw grid lines on the image with the visible grid
            cv2.rectangle(color_mapped_ndvi, (x, y), (x+dim_x-1, y+dim_y-1), (255, 255, 255), 1)
            image_with_visible_grid = color_mapped_ndvi

    return image_without_grid, image_with_visible_grid
    
def extract_vineyard_health_data(image_without_grid, vineyard_total_area_hectareas, outside_vineyard_color, ground_color):
    """
    Extracts vineyard health-related data from an image.

    Args:
        image_without_grid (numpy.ndarray): A numpy array representing the image without a visible grid.
        vineyard_total_area_hectares (float): Total area of the vineyard in hectares.

    Returns:
        tuple: A tuple containing the following values:
            - total_vineyard_pixels (int): Total pixels within the vineyard area.
            - total_plant_pixels (int): Total pixels representing plants within the vineyard area.
            - vine_area_meters (float): Vineyard area in square meters.
            - ndvi_mean_healthiness_level (float): Mean healthiness level based on pixel values.
            - percentage_ndvi_mean_level (float): Percentage representation of ndvi_mean_healthiness_level.
    """
    
    # Create a copy of the image to avoid modifying the original
    image_copy_without_grid = np.copy(image_without_grid)
    height, width = image_copy_without_grid.shape

    total_vineyard_pixels = 0 
    total_plant_pixels = 0
    healthiness_values = []

    # Iterate through each pixel in the image
    for y in tqdm(range(height)):
        for x in range(width):
            pixel_value = image_copy_without_grid[y,x]

            # Check if the pixel is outside the vineyard
            if pixel_value != outside_vineyard_color: 
                total_vineyard_pixels += 1 
            
            # Check if the pixel is within the vineyard but not ground
            if pixel_value != outside_vineyard_color and pixel_value != ground_color: 
                healthiness_values.append(image_copy_without_grid[y,x]) 
                total_plant_pixels += 1 

    # Calculate vineyard area in square meters
    vine_area_meters = (total_plant_pixels * vineyard_total_area_hectareas)/total_vineyard_pixels
    
    # Calculate the mean healthiness level based on pixel values
    ndvi_mean_healthiness_level = (np.sum(healthiness_values)/len(healthiness_values))/255
    
    # Calculate the percentage representation of ndvi_mean_healthiness_level
    percentage_ndvi_mean_level = np.interp(ndvi_mean_healthiness_level, [-1,1], [0,100])

    return total_vineyard_pixels, total_plant_pixels, vine_area_meters, ndvi_mean_healthiness_level, percentage_ndvi_mean_level
