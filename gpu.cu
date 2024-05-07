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
#define THREADS_PER_BLOCK 32
#define EPSILON 0.0001
#define MAX_ITERATIONS 100
#define CONVERGENCE_PERCENTAGE 80
const int K = 2;

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

__device__ __host__ float distance(float *x, float *y, int D)
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

/**
 * @brief update clusters centroids using reduction
 *
 * @param data_points_num
 * @param dimensions_num
 * @param clusters_num
 * @param data_points
 * @param centroids
 * @param cluster_assignment
 * @return __global__
 */
__global__ void update_cluster_centroids(int data_points_num, int dimensions_num,
                                         float *data_points, float *centroids, int *cluster_assignment, int *cluster_size)
{

    // thread in grid level
    const int grid_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (grid_tid == 0)
    {
        // print initial cluster sizes
        for (int i = 0; i < K; i++)
        {
            // printf("\n");
            printf(" FROM DEVICE Cluster %d size: %d\n", i, cluster_size[i]);
        }
    }
    // check for out of bounds
    if (grid_tid >= data_points_num)
        return;

    // thread index in block level
    const int block_tid = threadIdx.x;

    // Shared memory for reduction
    __shared__ float shared_data_points[THREADS_PER_BLOCK];

    // each thread loads the data point to shared memory
    shared_data_points[block_tid] = data_points[grid_tid];

    __shared__ float shared_cluster_assignment[THREADS_PER_BLOCK];

    // each thread loads the cluster assignment to shared memory
    shared_cluster_assignment[block_tid] = cluster_assignment[grid_tid];

    __syncthreads();

    if (block_tid == 0)
    {
        float data_point_sum[K] = {0};
        int temp_cluster_size[K] = {0};

        // for each data point, check its cluster assignment
        // and add the data point to the corresponding cluster
        for (int i = 0; i < blockDim.x; i++)
        {
            int cluster_id = shared_cluster_assignment[i];
            data_point_sum[cluster_id] += shared_data_points[i];
            temp_cluster_size[cluster_id] += 1;
        }

        // update the global centroids
        for (int i = 0; i < K; i++)
        {
            atomicAdd(&centroids[i], data_point_sum[i]);
            atomicAdd(&cluster_size[i], temp_cluster_size[i]);
        }
    }

    __syncthreads();

    // update the centroids
    if (grid_tid < K)
    {
        centroids[grid_tid] = centroids[grid_tid] / cluster_size[grid_tid];
    }

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    // Shared memory for reduction
    // __shared__ float s_centroids[32 * 3];
    // __shared__ int s_counts[32];

    // // Thread index in the block
    // int tid = threadIdx.x;

    // // Block index
    // int bid = blockIdx.x;

    // // Block size
    // int block_size = blockDim.x;

    // // Number of blocks
    // int num_blocks = gridDim.x;

    // // Initialize shared memory
    // for (int i = tid; i < clusters_num * dimensions_num; i += block_size)
    // {
    //     s_centroids[i] = 0;
    // }

    // for (int i = tid; i < clusters_num; i += block_size)
    // {
    //     s_counts[i] = 0;
    // }

    // __syncthreads();

    // // Update shared memory
    // for (int i = bid * block_size + tid; i < data_points_num; i += block_size * num_blocks)
    // {
    //     int cluster_id = cluster_assignment[i];
    //     for (int j = 0; j < dimensions_num; j++)
    //     {
    //         atomicAdd(&s_centroids[cluster_id * dimensions_num + j], data_points[i * dimensions_num + j]);
    //     }
    //     atomicAdd(&s_counts[cluster_id], 1);
    // }

    // __syncthreads();

    // // Reduction
    // for (int i = tid; i < clusters_num * dimensions_num; i += block_size)
    // {
    //     for (int j = 1; j < num_blocks; j++)
    //     {
    //         s_centroids[i] += s_centroids[j * block_size + i];
    //     }
    // }

    // for (int i = tid; i < clusters_num; i += block_size)
    // {
    //     for (int j = 1; j < num_blocks; j++)
    //     {
    //         s_counts[i] += s_counts[j * block_size + i];
    //     }
    // }

    // __syncthreads();

    // // Update centroids
    // for (int i = tid; i < clusters_num * dimensions_num; i += block_size)
    // {
    //     centroids[i] = s_centroids[i] / s_counts[i / dimensions_num];
    // }
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

__host__ bool check_convergence(float *centroids, float *new_centroids, int N, int D, int K)
{
    /*
    Function to check convergence

    args:
    centroids: centroids as a 1D array
    new_centroids: updated centroids as a 1D array
    N: number of data points
    D: number of dimensions
    K: number of clusters

    returns: True if converged, False otherwise
    */
    float centroids_distance = 0;

    // Compute distance between old and new centroids in all dimensions
    centroids_distance = distance(centroids, new_centroids, D);

    // printf("Centroids distance: %f\n", centroids_distance);
    if (centroids_distance < EPSILON)
    {
        return true;
    }
    return false;
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
    if (argc <= 2)
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

    int current_iteration = 0;

    // Number of blocks
    int N = width * height; // no of data points
    int D = channels;       // no of dimensions [1 as start]

    // Initialize centroids
    float *centroids = intilize_centroids(N, D, K, image);
    int *cluster_assignment = (int *)malloc(N * sizeof(int));

    // Device Memory Allocation
    float *d_image = 0;
    float *d_centroids = 0;
    int *d_cluster_assignment = 0;

    cudaMalloc(&d_image, N * D * sizeof(float));
    cudaMalloc(&d_centroids, K * D * sizeof(float));
    cudaMalloc(&d_cluster_assignment, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_image, image, N * D * sizeof(float), cudaMemcpyHostToDevice);
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    while (current_iteration < MAX_ITERATIONS)
    {
        // print the current
        current_iteration++;
        printf("Iteration: %d/%d\n", current_iteration, MAX_ITERATIONS);

        // Copy data from host to device
        cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

        // call the kernel [assign_data_points_to_centroids]
        assign_data_points_to_centroids<<<num_blocks, THREADS_PER_BLOCK>>>(N, D, K, d_image, d_centroids, d_cluster_assignment);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            // in red
            printf("\033[1;31m");
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            // reset color
            printf("\033[0m");
        }

        // Copy data from device to host
        cudaMemcpy(cluster_assignment, d_cluster_assignment, N * sizeof(int), cudaMemcpyDeviceToHost);

        int *cluster_size = (int *)malloc(K * sizeof(int));

        int *d_cluster_size = 0;
        cudaMalloc(&d_cluster_size, K * sizeof(int));

        float *updated_centroids = 0;
        cudaMalloc(&updated_centroids, K * D * sizeof(float));

        float *d_updated_centroids = 0;
        cudaMalloc(&d_updated_centroids, K * D * sizeof(float));

        update_cluster_centroids<<<num_blocks, THREADS_PER_BLOCK>>>(N, D, d_image, d_updated_centroids, d_cluster_assignment, d_cluster_size);
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            // in red
            printf("\033[1;31m");
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            // reset color
            printf("\033[0m");
        }

        // Copy data from device to host
        cudaMemcpy(updated_centroids, d_updated_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cluster_size, d_cluster_size, K * sizeof(int), cudaMemcpyDeviceToHost);

        // check convergence
        int convergedCentroids = 0;
        for (int i = 0; i < K; i++)
        {
            if (check_convergence(centroids + i * D, updated_centroids + i * D, N, D, K))
            {
                convergedCentroids++;
            }
        }
        printf("Converged Centroids: %d\n", convergedCentroids);
        // if 80% of the centroids have converged
        if (convergedCentroids >= K * CONVERGENCE_PERCENTAGE / 100.0)
        {
            printf("Converged\n");
            break;
        }

        // update the centroids
        centroids = updated_centroids;
    }

    // for (int i = 0; i < K; i++)
    // {
    //     printf("DEVICE Cluster %d size: %d\n", i, cluster_size[i]);
    // }

    // // print final centroids
    // for (int i = 0; i < K; i++)
    // {
    //     for (int j = 0; j < D; j++)
    //     {
    //         printf("aloooo %f ", centroids[i * D + j]);
    //     }
    //     printf("\n");
    // }

    printf("Cluster assignment done successfully :D\n");
    // for (int i = 0; i < N; i++)
    // {
    printf("Image: %f \n", image[0]);
    printf("Center: %d \n", cluster_assignment[0]);
    // }
}

// nvcc -o out_gpu_1  ./gpu.cu
// ./out_gpu_1 ./input.png 2

/**
 * @brief ZEBALA
 * int segment_start = blockIdx.x * blockDim.x * 2; // to load 2 data points per thread,
                                                     // ! later we'll need to consider data points dimensions

    int grid_tid = segment_start + threadIdx.x; // thread index in grid level
    int block_tid = threadIdx.x;                // thread index in block level
    // each thread is responsible for 2 data points
    for (int stride = blockDim.x; stride > 0; stride /= 2)
    {
        if (block_tid < stride)
        {
            data_points[grid_tid] += data_points[grid_tid + stride];
        }
        __syncthreads();
    }

    if (block_tid == 0)
    {
        centroids[blockIdx.x] = data_points[segment_start];
    }
 */