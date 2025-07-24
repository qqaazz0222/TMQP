import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

output_dir = "data/output"
patient_list = os.listdir(output_dir)
result_image = "calculate_graph.png"
result_image_list = []

for patient_id in patient_list:
    result_image_path = os.path.join(output_dir, patient_id, result_image)
    result_image_list.append(result_image_path)
    
result_image_list.sort()

# Load images from result_image_list
images = [Image.open(img_path) for img_path in result_image_list if os.path.exists(img_path)]

# Determine the size of each image
image_width, image_height = images[0].size if images else (0, 0)

# Define the grid dimensions
columns = 3
rows = (len(images) + columns - 1) // columns

# Create a blank canvas for the combined image
combined_width = columns * image_width
combined_height = rows * image_height
combined_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))

overlaid_width = image_width
overlaid_height = image_height
overlaid_image = Image.new("RGB", (overlaid_width, overlaid_height), (255, 255, 255))
overlaid_image_bg = Image.new("RGB", (overlaid_width, overlaid_height), (255, 255, 255))

# Paste each image into the combined image
for idx, img in enumerate(images):
    x_offset = (idx % columns) * image_width
    y_offset = (idx // columns) * image_height
    combined_image.paste(img, (x_offset, y_offset))

# Save the combined image
combined_image.save("combined_result.png")