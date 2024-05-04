import cv2
import sys

def save_as_grayscale(input_path, output_path):
    # Read the image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Convert the image to grayscale
    if image.shape[2] == 4:  # Check if image is four-channel (RGBA)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite(output_path, gray_image)

if __name__ == "__main__":
    # Check if command-line arguments are provided correctly
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_image_path> <output_image_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    save_as_grayscale(input_path, output_path)
