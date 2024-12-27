# CUDA K-Means Image Clustering

<p align="center">
  <img src="image.png" alt="K means Image" />
</p>

## <img  align= center width=80px src="giphy.gif">  Table of Content
1. [Overview](#Overview)
2. [Results](#Results)
3. [Get Started](#started)
4. [Contributors](#contributors)
5. [License](#license)


## <img align="center" width="50px" height="50px" src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/waving-hand_1f44b.gif"> Overview <a id="Overview"></a>

This tool takes an RGB image as input and applies clustering based on the number of centroids you specify. It also calculates the silhouette score to assess the quality of the clustering. Powered by CUDA parallelization, the process is optimized for high performance. :D

## <img align="center"  width =50px src="https://i.pinimg.com/originals/0c/2b/e6/0c2be68770c513163a18d46561c722cd.gif"> Results <a id = "Results"></a>

### Simple Image Low RESOLUTION

| ![alt text](./tests/image_3.png) | ![alt text](./tests/image_3_output_gpu_3_stream_0_sihouette.png) |
|----------------------------------|------------------------------------------------------------------|

### 4K Image

| ![alt text](./tests/high_res.jpg) | ![alt text](./tests/high_res_output_gpu_3_stream_0.png) |
|-----------------------------------|--------------------------------------------------------|

## <img  align= center width=50px height=50px src="https://cdn.pixabay.com/animation/2022/07/31/06/27/06-27-17-124_512.gif">Get Started <a id = "started"></a>

### Clustering
---------------------------------------------------------
### 1. Navigate to the Main Project Directory
```bash
cd ./
```

### 2. Compile Code
```bash
nvcc -o out_gpu_3_stream_0  ./src/gpu_3_stream_0.cu
```

### 3. Run the Code
Specify Parameters: `<path_to_image> <number_of_clusters>`  
Note: The number of clusters must be less than 20 😊
```bash
./out_gpu_3_stream_0 ./tests/image_3.png 5
```

### 4. Profile Results
```bash
nvprof ./out_gpu_3_stream_0 ./tests/image_3.png 5
```

### Clustering + Compute Silhouette Score
---------------------------------------------------------
### 1. Navigate to the Main Project Directory
```bash
cd ./
```

### 2. Compile Code
```bash
nvcc -o out_gpu_3_stream_0_sihouette  ./src/gpu_3_stream_0_sihouette.cu
```

### 3. Run Code
Specify Parameters: `<path_to_image> <number_of_clusters>`  
Note: The number of clusters must be less than 20 😊
```bash
./out_gpu_3_stream_0_sihouette ./tests/image_3.png 5
```

### 4. Profile Results
```bash
nvprof ./out_gpu_3_stream_0_sihouette ./tests/image_3.png 5
```

<!-- Contributors -->

## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/WFZvB7VIXBgiz3oDXE/giphy.gif?cid=6c09b952tmewuarqtlyfot8t8i0kh6ov6vrypnwdrihlsshb&rid=giphy.gif&ct=s"> Contributors <a id = "contributors"></a>

<!-- Contributors list -->
<table align="center" >
  <tr>
    <td align="center"><a href="https://github.com/BasmaElhoseny01"><img src="https://avatars.githubusercontent.com/u/72309546?v=4" width="150px;" alt=""/><br /><sub><b>Basma Elhoseny</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/SarahElzayat"><img src="https://avatars.githubusercontent.com/u/76779284?v=4" width="150px;" alt=""/><br /><sub><b>Sarah Elzayat</b></sub></a></td>

  </tr>
</table>

## <img  align= center height=50px src="https://moein.video/wp-content/uploads/2022/05/license-GIF-Certificate-Royalty-Free-Animated-Icon-350px-after-effects-project.gif">  License
This software is licensed under the [MIT License](https://github.com/BasmaElhoseny01/K-Means-Clustering-CUDA/blob/main/LICENSE). © Basma Elhoseny.