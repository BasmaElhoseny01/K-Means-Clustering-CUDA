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
int K = 2;
const int MAX_ITERATIONS = 100;
const float EPSILON = 1e-4;            // convergence threshold
const int CONVERGENCE_PERCENTAGE = 80; // Percentage of centroids that should converge to stop the algorithm

__host__ float *read_image(char *path, int *width, int *height, int *channels)
{

    // Read Image
    unsigned char *image_data = stbi_load(path, width, height, channels, 0);

    if (image_data == NULL)
    {
        printf("Error loading image\n");
        exit(1);
    }
    if (*channels != 1 && *channels != 3)
    {
        printf("Error: Image should be grayscale or RGB : %d\n", *channels);
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

__host__ float distance(float *x, float *y, int D)
{
    float dist = 0;
    for (int i = 0; i < D; i++)
    {
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(dist);
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

__host__ int *assign_data_points_to_centroids(int N, int D, int K, float *data_points, float *centroids)
{
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
    // Array to store cluster assignment for each data point [index of data point -> cluster number]
    int *cluster_assignment = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        float min_distance = FLT_MAX; // FLT_MAX represents the maximum finite floating-point value
        int min_centroid = -1;        // -1 represents no centroid
        for (int j = 0; j < K; j++)
        { // Compute distance between data point and centroid
            float dist = 0;
            dist = distance(data_points + i * D, centroids + j * D, D); // data_points[i * D] ,centroids[j * D]

            // Update min_distance and min_centroid
            if (dist < min_distance)
            {
                min_distance = dist;
                min_centroid = j;
            }
        }
        cluster_assignment[i] = min_centroid;
    }

    // printf("Cluster assignment done successfully :D\n");
    // for (int i = 0; i < N; i++)
    // {
    //     printf("%d ", cluster_assignment[i]);
    // }

    return cluster_assignment;
}

__host__ float *update_centroids(int N, int D, int K, float *data_points, float *centroids, int *cluster_assignment)
{
    /*
    Function to update the centroids

    args:
    N: number of data points
    D: number of dimensions
    K: number of clusters
    data_points: data points as a 1D array
    centroids: centroids as a 1D array
    cluster_assignment: cluster assignment for each data point

    returns: updated centroids
    */
    float *new_centroids = (float *)malloc(K * D * sizeof(float));

    // Initialize new_centroids to 0
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < D; j++)
        {
            new_centroids[i * D + j] = 0;
        }
    }

    // Count the number of data points in each cluster
    int *cluster_count = (int *)malloc(K * sizeof(int));
    for (int i = 0; i < K; i++)
    {
        cluster_count[i] = 0;
    }

    for (int i = 0; i < N; i++)
    {
        int cluster = cluster_assignment[i];
        for (int j = 0; j < D; j++)
        {
            new_centroids[cluster * D + j] += data_points[i * D + j];
        }
        cluster_count[cluster]++;
    }

    for (int i = 0; i < K; i++)
    {
        if (cluster_count[i] == 0)
        {
            printf("Warning: Empty cluster %d\n", i);
        }
    }
    // Update the centroids
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < D; j++)
        {
            new_centroids[i * D + j] /= cluster_count[i];
        }
    }

    // printf("*************************\n");
    // printf("Centroids updated successfully :D\n");
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

    return new_centroids;
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

__host__ float compute_intra_cluster_distance(int point_idx, float *data_points, int *data_points_assigments, int N, int D, int K)
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

__host__ float compute_inter_cluster_distance(int point_idx, float *data_points, int *data_points_assigments, int N, int D, int K)
{
    float nearest_centroid_dist = FLT_MAX;
    int nearest_centroid_idx = -1;

    // Compute distance between data_points[point_idx] and all other centroids :D to find the nearest centroid
    for (int i = 0; i < K; i++)
    {
        if (i != data_points_assigments[point_idx]) // Don't compute distance with the same cluster centroid :D
        {
            float dist = distance(data_points + point_idx * D, data_points + i * D, D);
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

__host__ float compute_shetollute_score(float *data_points, int *data_points_assigments, int N, int D, int K)
{
    /*
    Function to compute the shetollute value
    data_points: data points as a 1D array
    data_points_assigments: cluster assignment for each data point
    N: number of data points
    D: number of dimensions
    K: number of clusters


    args:
    data_points: data points as a 1D array
    */

    float shetollute_value = 0;

    for (int i = 0; i < N; i++)
    {
        float a_i = compute_intra_cluster_distance(i, data_points, data_points_assigments, N, D, K);
        float b_i = compute_inter_cluster_distance(i, data_points, data_points_assigments, N, D, K);

        shetollute_value += (b_i - a_i) / fmax(a_i, b_i);
    }

    return shetollute_value / N;
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

__host__ unsigned char *clutser_image(float *image, int width, int height, int channels, float *centroids)
{
    // Get assigned cluster for each pixel
    int N = width * height;
    int D = channels;
    int *cluster_assignment = assign_data_points_to_centroids(N, D, K, image, centroids);

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
    printf("Image clustered successfully :D\n");

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
    if (DEBUG)
    {

        printf("Hello World\n");
    }

    // Input Arguments
    if (argc != 3)
    {
        printf("Usage: %s <input_file>", argv[0]);
        exit(1);
    }

    char *input_file_path = argv[1];
    K = atoi(argv[2]);

    printf("Input file path: %s\n", input_file_path);

    // Read image
    int width, height, channels;
    float *image = read_image(input_file_path, &width, &height, &channels);

    int N = width * height; // no of data points
    int D = channels;       // no of dimensions [1 as start]

    // Initialize centroids
    float *centroids = intilize_centroids(N, D, K, image);

    int iteration = 0;

    // Compute Time
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    int *cluster_assignment;
    while (iteration < MAX_ITERATIONS)
    {
        iteration = iteration + 1;
        // printf("Iteration: %d/%d\n", iteration, MAX_ITERATIONS);

        // Assign each data point to the nearest centroid
        cluster_assignment = assign_data_points_to_centroids(N, D, K, image, centroids);

        // Update the centroids
        float *new_centroids = update_centroids(N, D, K, image, centroids, cluster_assignment);

        // printf("EPSILON: %f\n", EPSILON);
        int convergedCentroids = 0;
        for (int i = 0; i < K; i++)
        {
            if (check_convergence(centroids + i * D, new_centroids + i * D, N, D, K))
            {
                convergedCentroids++;
            }
        }
        // if 80% of the centroids have converged
        if (convergedCentroids >= K * CONVERGENCE_PERCENTAGE / 100.0)
        {
            break;
        }

        // Update centroids
        centroids = new_centroids;
    }

    // Compute Shilloute Score :D [Very Expensive]
    printf("Computing Shilloute Score ....\n");
    
    float shetollute_score = compute_shetollute_score(image, cluster_assignment, N, D, K);

    // Stop the timer
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Shetollute Score: %f\n", shetollute_score);
    if (DEBUG)
    {
        printf("Time taken [DEBUG]: %f sec\n", cpu_time_used);
    }
    else
    {
        printf("Time taken: %f sec\n", cpu_time_used);
    }

    if (!DEBUG)
    {
        printf("Converged after %d iterations\n", iteration);
    }
    if (iteration == MAX_ITERATIONS)
    {
        printf("Max Iterations reached :( \n");
    }

    // Cluster the image
    unsigned char *clutsered_image = clutser_image(image, width, height, channels, centroids);

    // Save the clustered image
    std::string input_path(input_file_path);
    std::string output_path = input_path.substr(0, input_path.find_last_of('.')) + "_output_cpu_3_silhouette.png";
    stbi_write_png(output_path.c_str(), width, height, 3, clutsered_image, width * 3);
    printf("Image saved successfully at: %s\n", output_path.c_str());

    return 0;
}

// nvcc -o out_cpu_3_silhouette   ./cpu_3_silhouette.cu
//   ./out_cpu_3_silhouette .\tests\image_3.png 5