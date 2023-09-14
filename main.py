""" Code designed to calculate vegetation indices and estimate the health of the vines using a grid"""

__author__ = "Paula Osés"
__copyright__ = "Copyright 2023, Noumena"
__credits__ = ["Paula Osés, Oriol Arroyo, Salvador Calgua, Aldo Sollazzo"]
__version__ = "1.0.1"
__maintainer__ = "Paula Osés"
__email__ = "paula@noumena.io"
__status__ = "Production"
__license__ = "MIT"

import cv2
import yaml
from src.utils import load_config, create_grid_with_color_mapping, extract_vineyard_health_data
from src.vegetation_indices import ndvi

def main():

    print()
    print(">>> Loading configuration file")

    # Load the configuration file
    config = load_config("config.yaml")

    # Load variables from the configuration file
    nir_image_filepath = config["nir_image_filepath"]
    rgb_image_filepath = config["rgb_image_filepath"]
    grid_image_filepath = config["grid_image_filepath"]
    grid_size_x = config["grid_size_x"]
    grid_size_y = config["grid_size_y"]
    vineyard_total_hectareas = config["vineyard_total_hectareas"]
    outside_vineyard_color = config["outside_vineyard_color"]
    ground_color = config["ground_color"]

    print()
    print(">>> Loading NIR and RGB images")

    # Load NIR and RGB orthomosaic images
    nir_image = cv2.imread(nir_image_filepath, cv2.IMREAD_GRAYSCALE)
    rgb_image = cv2.imread(rgb_image_filepath) # Apply white balance in preprocessing

    # Extract the red channel from the RGB image
    r_image = rgb_image[:,:,2] 

    # Check and resize images if necessary
    if nir_image.shape != r_image.shape:
        print("Image sizes are not equal")
        print("Resizing images...")
        height, width = nir_image.shape
        r = cv2.resize(r_image, (width, height))

    print()
    print(">>> Calculating Vegetation Indices")

    # Calculate vegetation indices (e.g., NDVI)
    ndvi_image = ndvi(nir_image, r_image)

    print()
    print(">>> Creating the grid and color mapping")

    # Create grid on the NDVI image
    image_without_grid, image_with_visible_grid = create_grid_with_color_mapping(ndvi_image, grid_size_x, grid_size_y)

    print()
    print(">>> Saving image with grid display at ",grid_image_filepath)

    # Save image
    cv2.imwrite(grid_image_filepath, image_with_visible_grid)

    print()
    print(">>> Extracting data related to vineyard health")

    # Extract data related to vineyard health
    total_vineyard_pixels, total_plant_pixels, vine_area_meters, ndvi_mean_healthiness_level, percentage_ndvi_mean_level = extract_vineyard_health_data(
                                                                                                                                    image_without_grid, vineyard_total_hectareas, 
                                                                                                                                    outside_vineyard_color, ground_color)
    # Print the extracted data
    print()    
    print("_______Vineyard health data_______")
    print("The total vineyard pixels are: ", total_vineyard_pixels)
    print("The total plant pixels are: ", total_plant_pixels)
    print("The vine area (in meters) is: ", vine_area_meters)
    print("The NDVI mean healthiness value is: ", ndvi_mean_healthiness_level)
    print("The NDVI mean value in percentage is: ", percentage_ndvi_mean_level)



if __name__ == "__main__":
    main()