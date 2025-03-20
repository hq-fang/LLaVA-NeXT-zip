import numpy as np  # For numerical operations and array handling
import cv2  # For drawing lines and circles on images
from PIL import Image  # For handling image input and output

# Additional context-specific imports
# Assuming `text_outputs` is a list of strings representing model output
# and `image` is a PIL Image object.

# Example text outputs
text_outputs = ['[(0.467, 0.985), (0.461, 0.953), (0.455, 0.92), (0.449, 0.885), (0.443, 0.849), (0.437, 0.812), (0.431, 0.774), (0.425, 0.735), (0.419, 0.696), (0.413, 0.656), (0.407, 0.616), (0.401, 0.576), (0.395, 0.536), (0.389, 0.497), (0.383, 0.459), (0.377, 0.422), (0.371, 0.386), (0.365, 0.351), (0.359, 0.318), (0.354, 0.287), (0.349, 0.26), (0.345, 0.236), (0.341, 0.215), (0.338, 0.197), (0.335, 0.182), (0.333, 0.17), (0.331, 0.161), (0.33, 0.156), (0.33, 0.155), (0.33, 0.154), (0.331, 0.151), (0.333, 0.145), (0.336, 0.136), (0.34, 0.126), (0.344, 0.116), (0.348, 0.106), (0.352, 0.096), (0.356, 0.086), (0.36, 0.076), (0.364, 0.066), (0.368, 0.056), (0.372, 0.046), (0.376, 0.036), (0.38, 0.026), (0.384, 0.016), (0.388, 0.006), (0.392, 0.001)]']

def extract_and_denormalize_coordinates(text_outputs, image_width, image_height):
    coordinates = []
    try:
        for output in text_outputs:
            if "[" in output and "]" in output:  # Check if there's a list of coordinates
                output = output.strip("[]")  # Remove brackets
                coord_pairs = output.split("), (")  # Split coordinate pairs
                for pair in coord_pairs:
                    x_str, y_str = pair.strip("()").split(", ")
                    x, y = float(x_str), float(y_str)
                    # Convert normalized coordinates to pixel coordinates
                    pixel_x = int(x * image_width)
                    # Flip vertically by subtracting y from 1
                    pixel_y = int((1 - y) * image_height)
                    coordinates.append((pixel_x, pixel_y))
    except ValueError as e:
        print(f"Error processing coordinates: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return coordinates




# Function to visualize the trajectory with smooth color transition
def draw_trajectory_with_color_transition(image, coordinates):
    start_color = np.array([255, 255, 255])  # White
    end_color = np.array([0, 255, 0])  # Green
    trajectory_len = len(coordinates)
    # Convert image to OpenCV format (BGR for OpenCV drawing)
    annotated_image = np.array(image)
    # Draw the trajectory with color transitions
    for i in range(1, trajectory_len):
        start_point = coordinates[i - 1]
        end_point = coordinates[i]
        # Calculate the color ratio for transition from white to green
        color_ratio = i / (trajectory_len - 1)
        line_color = (1 - color_ratio) * start_color + color_ratio * end_color
        line_color = tuple(map(int, line_color))  # Convert to integer
        # Draw the line with the calculated color
        cv2.line(annotated_image, start_point, end_point, line_color, 5)
    # Add a red dot at the last point in the trajectory
    cv2.circle(annotated_image, coordinates[-1], 10, (255, 0, 0), -1)  # Red dot
    # Convert image back to PIL format
    return Image.fromarray(annotated_image)
# Extract and denormalize coordinates
image = Image.open( "/net/nfs2.prior/jiafei/unified_VLM/datasets/slide_block_to_target/episode_37/front_0.png")

image_width, image_height = image.size
coordinates = extract_and_denormalize_coordinates(text_outputs, image_width, image_height)
# If coordinates exist, visualize and save the image
if coordinates:
    final_image = draw_trajectory_with_color_transition(image, coordinates)
    final_image.save('./output_image.png')  # Save the image with points drawn
    print("Image saved with trajectory.")
else:
    print("No coordinates found in model output.")