import numpy as np
import argparse
import cv2

from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt


# Function to generate distinct colors based on number of clusters
def generate_colors(num_clusters):
    colors = []
    for i in range(num_clusters):
        hue = int(360 * i / num_clusters)
        color = np.array([hue / 2, 255, 255], dtype=np.uint8)
        colors.append(cv2.cvtColor(np.array([[color]]), cv2.COLOR_HSV2RGB)[0, 0])
    return colors


def main():
    # Take image path from command
    parser = argparse.ArgumentParser(description="Sklearn K-means Clustering")

    parser.add_argument("image", type=str, help="Image path")
    parser.add_argument("K", type=int, help="Number of clusters")
     
    args = parser.parse_args()
    image_path = args.image
    # Choose number of clusters
    K=args.K
    
    # Read grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Flatten image
    data_points = image.flatten().reshape(-1, 1)


    # Apply K-means clustering
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(data_points)

    # Assign cluster labels to pixels
    cluster_labels = kmeans.labels_


    # Generate colors for clusters
    colors = generate_colors(K)


    # Replace pixel values with cluster colors
    clustered_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            clustered_image[i, j] = colors[cluster_labels[i * image.shape[1] + j]]

    # Display original and clustered image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(clustered_image)
    plt.title('Clustered Image')
    plt.show()

if __name__ == "__main__":
    main()

# python ./sklearn_kmeans.py ./tests/image_1_grey.png 3