# [Not WORKING] This script is used to crop white border from the image.
import cv2
import matplotlib.pyplot as plt
import argparse

def crop_white_border(image):
    # Convert image to grayscale
    # If image is already grayscale, this will return the same image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop image to bounding box
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

def main():
    # Take image path from command
    parser = argparse.ArgumentParser(description="Crop white border from image")
    # Image path
    parser.add_argument("image", type=str, help="Image path")

    args = parser.parse_args()
    image_path = args.image

    # Read image as it is
    image = cv2.imread(image_path)
    print("Original Image Shape: ", image.shape)

    # crop white square border frame for better clustering
    image_cropped = crop_white_border(image)
    print("Cropped Image Shape: ", image_cropped.shape)

    # Display the cropped image using mathplotlib
    # Display original and clustered image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(image_cropped)
    plt.title('Cropped Image')
    plt.show()

    
if __name__=="__main__":
    main()