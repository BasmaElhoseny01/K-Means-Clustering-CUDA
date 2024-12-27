#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <string.h>

__host__ float * read_input(const char *file_path, int &N, int &D, int &K)
{
    /*
    Function to read input file

    args:
    file_path: path to the input file
    N: number of data points
    D: number of dimensions
    K: number of clusters

    returns: data points as a 1D array
    */
    //   Open file
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        printf("Error: Unable to open file %s\n", file_path);
        exit(1);
    }
    printf("File opened successfully\n");


    fscanf(file, "%d", &N);
    fscanf(file, "%d", &D);
    fscanf(file, "%d", &K);

    // Print no of data points, dimensions and clusters
    printf("No of data points: %d\n", N);
    printf("No of dimensions: %d\n", D);
    printf("No of clusters: %d\n", K);

    // Read data points
    float *data_points = (float *)malloc(N * D * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
        {
            fscanf(file, "%f", &data_points[i * D + j]);//data_points[i][j]
        }
    }

    printf("Data points read successfully :D\n");

    // Close file
    fclose(file);

    return data_points;
}
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

    int N, D, K;

    // Read input file
    float *data_points = read_input(input_file_path, N, D, K);

}

// nvcc -o out  ./main.cu

// ./out ./input.txt