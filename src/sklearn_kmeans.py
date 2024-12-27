import numpy as np
import argparse
import cv2

from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score

SEED=42


# Function to generate distinct colors based on number of clusters
def generate_colors(num_clusters):
    colors = []
    for i in range(num_clusters):
        hue = int(360 * i / num_clusters)
        color = np.array([hue / 2, 255, 255], dtype=np.uint8)
        colors.append(cv2.cvtColor(np.array([[color]]), cv2.COLOR_HSV2RGB)[0, 0])
    return colors

def initialize_centroids(data_points, K):
    N, D = data_points.shape
    centroids = np.empty((K, D), dtype=np.float32)
    
    # Seed for randomization
    np.random.seed(SEED)  # Set your desired seed
    
    for i in range(K):
        # Each centroid is initialized to a random data point
        i_random = np.random.randint(N)
        centroids[i] = data_points[i_random]
    
    return centroids

def main():
    # Take image path from command
    parser = argparse.ArgumentParser(description="Sklearn K-means Clustering")

    parser.add_argument("image", type=str, help="Image path")
    parser.add_argument("K", type=int, help="Number of clusters")
     
    args = parser.parse_args()
    image_path = args.image
    # Choose number of clusters
    K=args.K
    
    # Read RGB image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0

    # Flatten image
    data_points = image.flatten().reshape(-1, 3)


    inti_centroids = [[0.078431,0.015686,0.000000],
                      [0.035294,0.000000,0.000000],
                      [0.164706,0.466667,0.184314],
                      [0.000000,0.011765,0.000000],
                      [0.294118,0.054902,0.082353]]


    # Apply K-means clustering
    kmeans = KMeans(n_clusters=K, init=inti_centroids, n_init=1, max_iter=100, random_state=SEED)
    kmeans.fit(data_points)

    # Assign cluster labels to pixels
    cluster_labels = kmeans.labels_

    # Print converged and number of iterations
    print("Converged:", kmeans.n_iter_)
    

    # Compute Silhouette Score
    silhouette_avg = silhouette_score(data_points, cluster_labels)
    print("Silhouette Score:", silhouette_avg)

    # Generate colors for clusters
    colors = generate_colors(K)


    # Replace pixel values with cluster colors
    clustered_image = np.zeros_like(image, dtype=np.uint8)  # Initialize with the same shape as the original image

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            clustered_image[i, j] = colors[cluster_labels[i * image.shape[1] + j]]

    # Display original and clustered image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)  # Assuming image is already in RGB format
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(clustered_image)
    plt.title('Clustered Image')
    plt.show()

if __name__ == "__main__":
    main()

# python ./sklearn_kmeans.py ./tests/image_1_grey.png 3