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

#define SEED 42

const int DEBUG = 0;

#define THREADS_PER_BLOCK 32
#define TITLEWIDTH 128
#define NUMOFSTREAMS 32
#define EPSILON 1e-4
#define MAX_ITERATIONS 100
#define CONVERGENCE_PERCENTAGE 80

const int K_max = 20;
const int D = 3;
int K = -1;

__host__ float *read_image(char *path, int *width, int *height, int *channels)
{

    // Read Image
    unsigned char *image_data = stbi_load(path, width, height, channels, 0);

    if (image_data == NULL)
    {
        printf("Error loading image\n");
        exit(1);
    }

    if (*channels != 3)
    {
        printf("Error: Image should be RGB : %d\n", *channels);
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

    if (true)
    {
        printf("Image loaded successfully\n");

        // printf("Width: %d, Height: %d, Channels: %d\n", *width, *height, *channels);
        // for (int i = 0; i < (*height) * (*width) * (*channels); i++)
        // {
        //     printf("%f ", image[i]);
        // }
    }

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
// Each thread in block its coreesponding centroid value [Colleased way]  (Small no of centroids os each thread load 1 dim of centroids)
__global__ void assign_data_points_to_centroids(int N, int D, int K, float *d_data_points, float *d_centroids, int *d_cluster_assignment)
{

    // Shared Mmeory for Centroids
    extern __shared__ float sh_centroids[];

    // thread index in block level
    const int block_tid = threadIdx.x;

    // thread index in grid level
    // each thread is responsible for 1 pixel
    const int grid_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Each Thread loads [Colleasing Way]
    // No of elements to be loaded by 1 thread
    int n_segments = (K * D + blockDim.x - 1) / blockDim.x; // ceil(K*D/THREADS_PER_BLOCK)

    for (int i = 0; i < n_segments; i++)
    {
        // Check Boundary
        if (block_tid + i * blockDim.x >= K * D)
        {
            break;
        }
        sh_centroids[block_tid + i * blockDim.x] = d_centroids[block_tid + i * blockDim.x];
    }

    // check for out of bounds
    // if the thread index is greater than the number of data points
    // then return
    if (grid_tid >= N)
        return;

    float min_distance = FLT_MAX; // FLT_MAX represents the maximum finite floating-point value
    int min_centroid = -1;        // -1 represents no centroid

    for (int i = 0; i < K; i++)
    {
        // Compute the distance between the data point and the centroids
        float dist = distance(d_data_points + grid_tid * D, sh_centroids + i * D, D);

        if (dist < min_distance)
        {
            min_distance = dist;
            min_centroid = i;
        }
    }

    // Assign the data point to the nearest centroid
    d_cluster_assignment[grid_tid] = min_centroid;

    // Distance of the data point to the nearest centroid
    // d_distances[grid_tid] = min_distance;
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
                                         float *d_data_points, int *d_cluster_assignment, float *d_centroids, int *d_cluster_sizes, int K)
{
    // thread in grid level
    const int grid_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // check for out of bounds
    if (grid_tid >= data_points_num)
        return;

    // thread index in block level
    const int block_tid = threadIdx.x;

    // Shared Memory
    __shared__ float sh_data_point_sum[K_max * D]; // sum of data points for each cluster
    __shared__ int sh_cluster_size[K_max];         // temporary cluster size

    // Initialize shared memory array
    if (block_tid < K * D)
    {
        sh_data_point_sum[block_tid] = 0.0;
        if (block_tid < K)
        {
            sh_cluster_size[block_tid] = 0.0;
        }
    }
    __syncthreads();

    const int data_point_assignment = d_cluster_assignment[grid_tid];
    atomicAdd(&sh_cluster_size[data_point_assignment], 1);

    for (int j = 0; j < dimensions_num; j++)
    {
        atomicAdd(&sh_data_point_sum[data_point_assignment * dimensions_num + j], d_data_points[grid_tid * dimensions_num + j]);
    }

    __syncthreads();

    // Add to the Global Memory
    // update the global centroids
    // if (threadIdx.x == 0)
    // {
    //     for (int i = 0; i < K; i++)
    //     {
    //         atomicAdd(&d_cluster_sizes[i], sh_cluster_size[i]);
    //         for (int j = 0; j < dimensions_num; j++)
    //         {
    //             atomicAdd(&d_centroids[i * dimensions_num + j], sh_data_point_sum[i * dimensions_num + j]);
    //         }
    //     }
    // }

    if (threadIdx.x < K * dimensions_num)
    {
        if (threadIdx.x < K)
        {
            atomicAdd(&d_cluster_sizes[threadIdx.x], sh_cluster_size[threadIdx.x]);
        }
        for (int j = 0; j < dimensions_num; j++)
        {
            atomicAdd(&d_centroids[threadIdx.x * dimensions_num + j], sh_data_point_sum[threadIdx.x * dimensions_num + j]);
        }
    }
}

__device__ __host__ float compute_intra_cluster_distance(int point_idx, float *data_points, int *data_points_assigments, int N, int D, int K)
{
    /*
    Function to compute the intra cluster distance

    args:
    point_idx: index of the data point to compute the intra cluster distance
    data_points: data points as a 1D array
    data_points_assigments: cluster assignment for each data point
    N: number of data points
    D: number of dimensions
    K: number of clusters

    returns: intra cluster distance from point_idx to all other points in the same cluster
    */

    float intra_cluster_distance = 0;
    int count = 0;

    for (int i = 0; i < N; i++)
    {
        if (data_points_assigments[i] == data_points_assigments[point_idx] && i != point_idx) // Don't compute distance with itself :D
        {
            // In the same cluster
            // Compute distance between data_points[i] and data_points[point_idx]
            intra_cluster_distance += distance(data_points + i * D, data_points + point_idx * D, D);
            count++;
        }
    }

    // Average distance :D
    return (count == 0) ? 0.0 : intra_cluster_distance / count;
}

__device__ __host__ float compute_inter_cluster_distance(int point_idx, float *data_points, int *data_points_assigments, float *d_centroids, int N, int D, int K)
{
    float nearest_centroid_dist = FLT_MAX;
    int nearest_centroid_idx = -1;

    // Compute distance between data_points[point_idx] and all other centroids :D to find the nearest centroid
    for (int i = 0; i < K; i++)
    {
        if (i != data_points_assigments[point_idx]) // Don't compute distance with the same cluster centroid :D
        {
            float dist = distance(data_points + point_idx * D, d_centroids + i * D, D);
            if (dist < nearest_centroid_dist)
            {
                nearest_centroid_idx = i;
            }
        }
    }

    float inter_cluster_distance = 0;
    int count = 0;

    // Compute distance between data_points[point_idx] and all other points in the nearest centroid cluster
    for (int i = 0; i < N; i++)
    {
        if (data_points_assigments[i] == nearest_centroid_idx)
        {
            // In the same cluster
            // Compute distance between data_points[i] and data_points[point_idx]
            inter_cluster_distance += distance(data_points + i * D, data_points + point_idx * D, D);
            count++;
        }
    }

    // Average distance :D
    return (count == 0) ? 0.0 : inter_cluster_distance / count;
}

/*
Function to compute the shilloute score for all data points

args:
data_points: data points as a 1D array
data_points_assigments: cluster assignment for each data point
N: number of data points
D: number of dimensions
K: number of clusters

returns: shilloute score
*/
__global__ void compute_shetollute_score(float *d_data_points, int *d_cluster_assignment, float *d_centroids, int N, int D, int K, float *d_shilloute_scores)
{

    // Each thread is responsible for 1 element in the Title
    // No of elements has to be computed by 1 thread
    int n_elements_per_thread = (TITLEWIDTH + blockDim.x - 1) / blockDim.x; // ceil(TITLEWIDTH/THREADS_PER_BLOCK)

    // thread in grid level
    const int grid_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // thread index in block level
    const int block_tid = threadIdx.x;

    // if (grid_tid==0)
    // {
    //     printf("no of elements per thread: %d\n", n_elements_per_thread);
    // }

    __shared__ float sh_shilloute_scores[TITLEWIDTH];

    // Start of the Title in the data points
    // Data Points |--------------TITLEWIDTH----------------|--------TITLEWIDTH------------------
    int start = blockIdx.x * TITLEWIDTH;

    // In the block level each thread is responsible for n_elements_per_thread element in the Title
    for (int i = 0; i < n_elements_per_thread; i++)
    {
        // |*-----THREADSPERBLOCK---*---THREADSPERBLOCK---*---THREADSPERBLOCK----*---THREADSPERBLOCK------|--------TITLEWIDTH------------------
        // Check for out of bounds
        if (start + block_tid + i * blockDim.x >= N)
        {
            // Add 0 to the shared memory
            // d_shilloute_scores[start + block_tid + i * blockDim.x] = 0;
            return;
        }

        // Compute the average distance of the data point to all other points in the same cluster
        float a = compute_intra_cluster_distance(start + block_tid + i * blockDim.x, d_data_points, d_cluster_assignment, N, D, K);
        // Compute the average distance of the data point to all other points in the nearest cluster
        float b = compute_inter_cluster_distance(start + block_tid + i * blockDim.x, d_data_points, d_cluster_assignment, d_centroids, N, D, K);
        float shilloute_score = (b - a) / max(a, b);

        // if (block_tid == 0)
        // {
        // printf("Data Point: %d, a: %f, b: %f, shilloute_score: %f\n", grid_tid, a, b, shilloute_score);
        // }

        // Store the shilloute score in shared memory
        // d_shilloute_scores[start + block_tid + i * blockDim.x] = shilloute_score;
        sh_shilloute_scores[block_tid + i * blockDim.x] = shilloute_score;
    }
    __syncthreads();

    // Compute the sum of shilloute scores in the block
    if (block_tid == 0)
    {
        // Compute the sum of shilloute scores in the block
        float sum = 0;
        for (int i = 0; i < TITLEWIDTH; i++)
        {
            sum += sh_shilloute_scores[i];
        }
        d_shilloute_scores[blockIdx.x] = sum;
    }

    // Apply Reduction to Compute the Sum of Shilloute Scores

    // Computuation
    // for (int i=0; i<n_elements_per_thread; i++)

    // __shared__ float sh_shilloute_scores[THREADS_PER_BLOCK];

    // // check for out of bounds
    // if (grid_tid >= N)
    // {
    //     // Add 0 to the shared memory
    //     sh_shilloute_scores[block_tid] = 0;
    //     return;
    // }

    // // Compute the average distance of the data point to all other points in the same cluster
    // float a = compute_intra_cluster_distance(grid_tid, d_data_points, d_cluster_assignment, N, D, K);
    // // Compute the average distance of the data point to all other points in the nearest cluster
    // float b = compute_inter_cluster_distance(grid_tid, d_data_points, d_cluster_assignment, d_centroids, N, D, K);
    // float shilloute_score = (b - a) / max(a, b);

    // // if (block_tid == 0)
    // // {
    // //     printf("Data Point: %d, a: %f, b: %f, shilloute_score: %f\n", grid_tid, a, b, shilloute_score);
    // // }

    // // Store the shilloute score in shared memory
    // sh_shilloute_scores[block_tid] = shilloute_score;
    // // d_shilloute_scores_sum[grid_tid] = shilloute_score;

    // __syncthreads();

    // // Compute the sum of shilloute scores in the block
    // // Reduction Sum
    // for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    // {
    //     if (block_tid < stride)
    //     {
    //         sh_shilloute_scores[block_tid] += sh_shilloute_scores[block_tid + stride];
    //     }
    //     __syncthreads();
    // }

    // // Store the sum of shilloute scores in the global memory
    // if (block_tid == 0)
    // {
    //     d_shilloute_scores_sum[blockIdx.x] = sh_shilloute_scores[0];
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
    // srand(time(NULL)); // Seed for randomization
    srand(SEED); // Seed for randomization

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

    if (true)
    {
        printf("Centroids initialized successfully :D\n");

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < D; j++)
            {
                printf("%f ", centroids[i * D + j]);
            }
            printf("\n");
        }
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

__host__ int **generate_cluster_color()
{
    int **cluster_color = (int **)malloc(K * sizeof(int *));

    for (int i = 0; i < K; i++)
    {
        cluster_color[i] = (int *)malloc(3 * sizeof(int));
        for (int j = 0; j < 3; j++)
        {
            cluster_color[i][j] = rand() % 256;
        }
        // if reepating colors [Reassign color]
        for (int j = 0; j < i; j++)
        {
            if (cluster_color[i][0] == cluster_color[j][0] && cluster_color[i][1] == cluster_color[j][1] && cluster_color[i][2] == cluster_color[j][2])
            {
                i--;
                break;
            }
        }
    }
    return cluster_color;
}

__host__ unsigned char *clutser_image(float *image, int width, int height, int *cluster_assignment)
{
    // Get assigned cluster for each pixel
    int N = width * height;

    // Cluster the image
    unsigned char *clustered_image = (unsigned char *)malloc(sizeof(unsigned char) * height * width * 3);
    // float *clustered_image = (float *)malloc(sizeof(float) * height * width * 3);

    // Generate Cluster Colors
    int **cluster_color = generate_cluster_color();
    if (DEBUG)
    {

        printf("Cluster Colors Generated\n");
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                printf("%d ", cluster_color[i][j]);
            }
            printf("\n");
        }
    }

    // Assign color to each pixel
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            clustered_image[i * 3 + j] = cluster_color[cluster_assignment[i]][j];
        }
    }

    if (DEBUG)
    {
        printf("Image clustered successfully :D\n");
    }

    // Write Image
    return clustered_image;
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
    if (argc < 2)
    {
        printf("Usage: %s <input_file> <K>", argv[0]);
        exit(1);
    }

    char *input_file_path = argv[1];
    K = atoi(argv[2]);

    if (true)
    {
        printf("Input file path: %s\n", input_file_path);
    }

    // Read image
    int width, height, channels;
    float *image = read_image(input_file_path, &width, &height, &channels);

    // Number of blocks
    int N = width * height; // no of data points

    // Initialize centroids
    float *centroids = intilize_centroids(N, D, K, image);
    int *cluster_assignment = (int *)malloc(N * sizeof(int));
    int *cluster_sizes = (int *)malloc(K * sizeof(int));          // Array to store the size of each cluster
    float *shilloute_scores = (float *)malloc(N * sizeof(float)); // Array to store the size of each cluster

    // Device Memory Allocation
    float *d_image = 0;
    float *d_centroids = 0;
    int *d_cluster_assignment = 0;
    int *d_cluster_sizes = 0;
    float *d_shilloute_scores = 0;

    cudaMalloc(&d_image, N * D * sizeof(float));
    cudaMalloc(&d_centroids, K * D * sizeof(float));
    cudaMalloc(&d_cluster_assignment, N * sizeof(int));
    cudaMalloc(&d_cluster_sizes, K * sizeof(int));      // Array to store the size of each cluster
    cudaMalloc(&d_shilloute_scores, N * sizeof(float)); // Array to store the size of each cluster

    // Compute Time
    clock_t start, end;
    double total_time;
    start = clock();

    // Start Streaming for First Iteration
    // Create streams
    cudaStream_t streams[NUMOFSTREAMS];
    for (int i = 0; i < NUMOFSTREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    int numSegments = NUMOFSTREAMS;                        // Data Divided into count of streams
    int segmentSize = (N + numSegments - 1) / numSegments; // Size of each segment  ceil(N/numSegments)

    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // ceil(N/THREADS_PER_BLOCK)

    // Copy data from host to device [centroids]
    cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

    // Streams loop
    for (int s = 0; s < numSegments; s++)
    {
        int start = s * segmentSize;
        int end = min(start + segmentSize, N); // min to handle the last segment
        int Nsegment = end - start;

        cudaMemcpyAsync(d_image + start * D, image + start * D, Nsegment * D * sizeof(float), cudaMemcpyHostToDevice, streams[s]);

        // call the kernel [assign_data_points_to_centroids]
        assign_data_points_to_centroids<<<num_blocks, THREADS_PER_BLOCK, K * D * sizeof(float), streams[s]>>>(Nsegment, D, K, d_image + start, d_centroids, d_cluster_assignment + start);
    }

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // in red
        printf("\033[1;31m");
        printf("CUDA error [After assign_data_points_to_centroids()]: %s\n", cudaGetErrorString(error));
        // reset color
        printf("\033[0m");
    }

    if (DEBUG)
    {
        printf("Cluster assignment done successfully :D\n");
        // cudaMemcpy(cluster_assignment, d_cluster_assignment, N * sizeof(int), cudaMemcpyDeviceToHost); // [FOR DEGUB]
        // for (int i = 0; i < N; i++)
        // {
        //     printf("%d ", cluster_assignment[i]);
        // }
    }
    int iteration = 0;
    while (iteration < MAX_ITERATIONS)
    {
        // print the current
        iteration++;

        if (DEBUG)
        {
            printf("Iteration: %d/%d\n", iteration, MAX_ITERATIONS);
        }

        // Reset the cluster sizes
        cudaMemset(d_cluster_sizes, 0, K * sizeof(int));
        cudaMemset(d_centroids, 0, K * D * sizeof(float)); // I see it is very important to reset the centroids but removing it will not affect the results ??!!!

        update_cluster_centroids<<<num_blocks, THREADS_PER_BLOCK>>>(N, D, d_image, d_cluster_assignment, d_centroids, d_cluster_sizes, K);
        cudaDeviceSynchronize();
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            // in red
            printf("\033[1;31m");
            printf("CUDA error [After update_cluster_centroids()]: %s\n", cudaGetErrorString(error));
            // reset color
            printf("\033[0m");
        }

        // Copy data from device to host
        // To Hold new Centroids
        float *new_centroids = (float *)malloc(K * D * sizeof(float));
        cudaMemcpy(new_centroids, d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cluster_sizes, d_cluster_sizes, K * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < K; i++)
        {
            if (cluster_sizes[i] == 0)
            {
                printf("Warning: Empty cluster %d\n", i);
            }
        }

        // Update the centroids
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < D; j++)
            {
                new_centroids[i * D + j] /= cluster_sizes[i];
            }
        }

        if (DEBUG)
        {
            printf("Centroids updated successfully :D\n");
            // printf("*************************\n");
            // // Print old and new centroids
            // printf("Old Centroids\n");
            // for (int i = 0; i < K; i++)
            // {
            //     for (int j = 0; j < D; j++)
            //     {
            //         printf("%f ", centroids[i * D + j]);
            //     }
            //     printf("\n");
            // }
            // printf("\nNew Centroids\n");
            // for (int i = 0; i < K; i++)
            // {
            //     for (int j = 0; j < D; j++)
            //     {
            //         printf("%f ", new_centroids[i * D + j]);
            //     }
            //     printf("\n");
            // }
            // printf("*************************\n");
        }

        // check convergence
        int convergedCentroids = 0;
        for (int i = 0; i < K; i++)
        {
            if (check_convergence(centroids + i * D, new_centroids + i * D, N, D, K))
            {
                convergedCentroids++;
            }
        }
        if (DEBUG)
        {
            printf("Converged Centroids: %d\n", convergedCentroids);
        }
        // if 80% of the centroids have converged
        if (convergedCentroids >= K * CONVERGENCE_PERCENTAGE / 100.0)
        {

            if (DEBUG)
            {
                printf("Converged after %d iterations\n", iteration);
            }
            break;
        }

        // Update centroids
        centroids = new_centroids;

        // Copy data from host to device [centroids]
        cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

        // call the kernel [assign_data_points_to_centroids]
        assign_data_points_to_centroids<<<num_blocks, THREADS_PER_BLOCK, K * D * sizeof(float)>>>(N, D, K, d_image, d_centroids, d_cluster_assignment);

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            // in red
            printf("\033[1;31m");
            printf("CUDA error [After assign_data_points_to_centroids()]: %s\n", cudaGetErrorString(error));
            // reset color
            printf("\033[0m");
        }

        if (DEBUG)
        {
            printf("Cluster assignment done successfully :D\n");
            // cudaMemcpy(cluster_assignment, d_cluster_assignment, N * sizeof(int), cudaMemcpyDeviceToHost); // [FOR DEGUB]
            // for (int i = 0; i < N; i++)
            // {
            //     printf("%d ", cluster_assignment[i]);
            // }
        }
    }

    // Compute Shilloute Score :D [Very Expensive]
    // if (DEBUG)
    if (true)
    {
        printf("Computing Shilloute Score ....\n");
    }

    int num_blocks_k3 = (N + TITLEWIDTH - 1) / TITLEWIDTH; // ceil(N/THREADS_PER_BLOCK)
    // printf("N: %d\n", N);
    // printf("Title Width: %d\n", TITLEWIDTH);
    // printf("Number of Threads Per Block: %d\n", THREADS_PER_BLOCK);
    // printf("Number of Blocks: %d\n", num_blocks);

    // // Make cluter assignment random
    // for (int i = 0; i < N; i++)
    // {
    //     cluster_assignment[i] = rand() % K;
    // }
    // Copy to device
    // cudaMemcpy(d_cluster_assignment, cluster_assignment, N * sizeof(int), cudaMemcpyHostToDevice);
    compute_shetollute_score<<<num_blocks_k3, THREADS_PER_BLOCK>>>(d_image, d_cluster_assignment, d_centroids, N, D, K, d_shilloute_scores);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // in red
        printf("\033[1;31m");
        printf("CUDA error [After compute_shetollute_score()]: %s\n", cudaGetErrorString(error));
        // reset color
        printf("\033[0m");
    }

    // Copy Shilloute Scores To Host
    cudaMemcpy(shilloute_scores, d_shilloute_scores, num_blocks * sizeof(float), cudaMemcpyDeviceToHost); // Copy no of blocks only :D

    // Compute the average shilloute score
    float shetollute_score = 0;
    for (int i = 0; i < num_blocks; i++)
    {
        // printf("Shilloute Score: %f\n", shilloute_scores[i]);
        shetollute_score += shilloute_scores[i];
    }
    shetollute_score /= N;

    // Stop the timer
    end = clock();
    total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Shetollute Score: %f\n", shetollute_score);
    if (DEBUG)
    {
        printf("Time taken [DEBUG]: %f sec\n", total_time);
    }
    else
    {
        printf("Time taken: %f sec\n", total_time);
    }

    if (!DEBUG)
    {
        printf("Converged after %d iterations\n", iteration);
    }

    if (iteration == MAX_ITERATIONS)
    {
        printf("Max Iterations reached :( \n");
    }

    // Copy Assigments
    cudaMemcpy(cluster_assignment, d_cluster_assignment, N * sizeof(int), cudaMemcpyDeviceToHost);

    // // Cluster Assignments
    // printf("*************************\n");
    // for (int i = 0; i < N; i++)
    // {
    // printf("%d ", cluster_assignment[i]);
    // }

    // Cluster the image
    unsigned char *clutsered_image = clutser_image(image, width, height, cluster_assignment);

    // Save the clustered image
    std::string input_path(input_file_path);
    std::string output_path = input_path.substr(0, input_path.find_last_of('.')) + "_output_gpu_3_stream_0_sihouette.png";
    stbi_write_png(output_path.c_str(), width, height, 3, clutsered_image, width * 3);
    printf("Image saved successfully at: %s\n", output_path.c_str());

    return 0;
}

// nvcc -o out_gpu_3_stream_0_sihouette_2_mul_1  ./gpu_3_stream_0_sihouette_2_mul_1.cu
// ./out_gpu_3_stream_0_sihouette_2_mul_1 .\tests\image_3.png 5
// D:\Parallel-Computing-Project>nvprof -o ./profiles/out_gpu_3_stream_0_sihouette_2_mul_1.nvprof ./out_gpu_3_stream_0_sihouette_2_mul_1 ./tests/image_3.png 5
