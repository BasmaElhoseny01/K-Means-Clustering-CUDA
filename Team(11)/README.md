# CUDA Kmeans Image Clustering

## <img align="center"  width =50px  height =50px src="https://em-content.zobj.net/source/animated-noto-color-emoji/356/waving-hand_1f44b.gif"> Overview <a id = "Overview"></a>

Given a RGB Image, You will get a clustered image with the no of centroids you specify and you can get silhouette score to see accuracy of our clustering techniques using CUDA paralyzation :D

## Results

### Simple Image Low RESOLUTION

![alt text](../tests/image_3.png)
![alt text](../tests/image_3_output_gpu_3_stream_0_sihouette.png)

### 4K Image

![alt text](high_res.jpg)
![alt text](high_res_output_gpu_3_stream_0.png)

## <img  align= center width=50px height=50px src="https://cdn.pixabay.com/animation/2022/07/31/06/27/06-27-17-124_512.gif">Get Started <a id = "started"></a>

### Kmeans Clustering For Image

##### Go to the main directory of the project

```
cd ./
```

##### Compile Code

```
nvcc -o out_gpu_3_stream_0  ./gpu_3_stream_0.cu
```

#### Run Code

##### Specify Parameters <path_to_image> <no_of_clusters>

no of clusters must be less than 20 😊

```
./out_gpu_3_stream_0 .\tests\image_3.png 5
```

#### Profile Results

```
nvprof ./out_gpu_3_stream_0 ./tests/image_3.png 5
```

### Kmeans Clustering + Compute Silhouette Score

#### Compile Code

```
nvcc -o out_gpu_3_stream_0_sihouette  ./gpu_3_stream_0_sihouette.cu
```

#### Run Code

##### Specify Parameters <path_to_image> <no_of_clusters>

no of clusters must be less than 20 😊

```
./out_gpu_3_stream_0_sihouette .\tests\image_3.png 5
```

#### Profile Results

```
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
