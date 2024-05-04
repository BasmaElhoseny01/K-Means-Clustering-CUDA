#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int K = 2;

__host__ float *read_image(char *path, int *width, int *height, int *channels)
{

    // Read Image
    unsigned char *image_data = stbi_load(path, width, height, channels, 0);

    if (image_data == NULL)
    {
        printf("Error loading image\n");
        exit(1);
    }
    if (*channels != 1)
    {
        printf("Error: Image should be grayscale: %d\n", *channels);
        exit(1);
    }

    // Host Memory Allocation & convert data from unsigned char to float
    float *image = (float *)malloc(sizeof(float) * (*width) * (*height) * (*channels));

    // Normlaization
    for (int i = 0; i < (*height) * (*width) * (*channels); i++)
    {
        image[i] = (float)image_data[i] / 255.0f;
    }

    if (*image == NULL)
    {
        printf("Error loading image\n");
        exit(1);
    }

    // Free the loaded image
    stbi_image_free(image_data);

    printf("Image loaded successfully\n");

    // for (int i = 0; i < (*height) * (*width) * (*channels); i++)
    // {
    //     printf("%f ", image[i]);
    // }

    return image;
}

__device__ float distance(float *x, float *y, int D)
{
    float dist = 0;
    for (int i = 0; i < D; i++)
    {
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(dist);
}

// __global__ void kMeansClusterAssignment(float *d_datapoints, int *d_clust_assn, float *d_centroids)
// {
// 	//get idx for this datapoint
// 	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

// 	//bounds check
// 	if (idx >= N) return;

// 	//find the closest centroid to this datapoint
// 	float min_dist = INFINITY;
// 	int closest_centroid = 0;

// 	for(int c = 0; c<K;++c)
// 	{
// 		float dist = distance(d_datapoints[idx],d_centroids[c]);

// 		if(dist < min_dist)
// 		{
// 			min_dist = dist;
// 			closest_centroid=c;
// 		}
// 	}

// 	//assign closest cluster id for this datapoint/thread
// 	d_clust_assn[idx]=closest_centroid;
// }

/*
Function to assign each data point to the nearest centroid

args:
N: number of data points
D: number of dimensions
K: number of clusters
data_points: data points as a 1D array
centroids: centroids as a 1D array
cluster_assignment: cluster assignment for each data point

returns: None
*/
__global__ void assign_data_points_to_centroids(int N, int D, int K, float *data_points, float *centroids, int *cluster_assignment)
{
    // thread index in grid level
    // each thread is responsible for 1 pixel
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // check for out of bounds
    // if the thread index is greater than the number of data points
    // then return
    if (tid >= N)
        return;

    float min_distance = FLT_MAX; // FLT_MAX represents the maximum finite floating-point value
    int min_centroid = -1;        // -1 represents no centroid

    for (int i = 0; i < K; i++)
    {
        // Compute the distance between the data point and the centroid
        float dist = distance(data_points + tid * D, centroids + i * D, D);

        if (dist < min_distance)
        {
            min_distance = dist;
            min_centroid = i;
        }
    }

    // Assign the data point to the nearest centroid
    cluster_assignment[tid] = min_centroid;
}

__host__ float *intilize_centroids(int N, int D, int K, float *data_points)
{
    /*
    Function to initialize centroids randomly as one of the data points

    args:
    N: number of data points
    D: number of dimensions
    K: number of clusters
    data_points: data points as a 1D array

    returns: centroids as a 1D array
    */
    srand(time(NULL)); // Seed for randomization

    float *centroids = (float *)malloc(K * D * sizeof(float));
    for (int i = 0; i < K; i++)
    {
        // Each centroid is initialized to a Random data point
        int i_random = rand() % N;
        for (int j = 0; j < D; j++)
        {
            centroids[i * D + j] = data_points[i_random * D + j];
        }
    }

    printf("Centroids initialized successfully :D\n");

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < D; j++)
        {
            printf("%f ", centroids[i * D + j]);
        }
        printf("\n");
    }

    return centroids;
}
/*
Kmeans:
1. Initialize centroids (Random or Kmeans++)
2. Assign each data point to the nearest centroid
3. Update the centroids
4. Repeat 2 and 3 until convergence
*/
int main(int argc, char *argv[])
{

    printf("Hello World\n");

    // Input Arguments
    if (argc != 2)
    {
        printf("Usage: %s <input_file>", argv[0]);
        exit(1);
    }

    char *input_file_path = argv[1];
    // K = atoi(argv[2]);

    printf("Input file path: %s\n", input_file_path);

    // Read image
    int width, height, channels;
    float *image = read_image(input_file_path, &width, &height, &channels);

    int N = width * height; // no of data points
    int D = channels;       // no of dimensions [1 as start]

    // Initialize centroids
    float *centroids = intilize_centroids(N, D, K, image);
    int *cluster_assignment = (int *)malloc(N * sizeof(int));

    // Device Memory Allocation
    float *d_image;
    float *d_centroids;
    int *d_cluster_assignment;

    cudaMalloc(&d_image, N * D * sizeof(float));
    cudaMalloc(&d_centroids, K * D * sizeof(float));
    cudaMalloc(&d_cluster_assignment, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_image, image, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

    // Number of threads per block
    int num_threads = 32;

    // Number of blocks
    int num_blocks = (N + num_threads - 1) / num_threads;

    // call the kernel
    assign_data_points_to_centroids<<<num_blocks, num_threads>>>(N, D, K, d_image, d_centroids, d_cluster_assignment);

    // Copy data from device to host
    cudaMemcpy(cluster_assignment, d_cluster_assignment, N * sizeof(int), cudaMemcpyDeviceToHost);


    printf("Cluster assignment done successfully :D\n");
    // for (int i = 0; i < N; i++)
    // {
    printf("Image: %f \n", image[0]);
    printf("Center: %d \n", cluster_assignment[0]);
    // }
}

// nvcc -o out_gpu_1  ./gpu.cu
// ./out_gpu_1 ./input.png 2