#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

cv::Mat loadImage(){
    cv::Mat image = cv::imread("Imagen/BananaCat.jpg", cv::IMREAD_COLOR);

    if (image.empty())
    {
        std::cerr << "No se pudo abrir o encontrar la imagen" << std::endl;
        exit(1);
    }

    return image;
}

__global__ void imageToGreyScale(uchar3 *rgb, uchar *grey, int rows, int cols)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows)
    {
        int rgb_offset = ((row * cols) + col) * 3;

        uchar b = rgb[rgb_offset].x;
        uchar g = rgb[rgb_offset].y;
        uchar r = rgb[rgb_offset].z;

        grey[row * cols + col] = r * 0.21f + g * 0.71f + b * 0.07f;
    }
}

/**
 * Utilidad para obtener la hora en segundos.
 *
 * @returns El tiempo actual del cpu en segundos.
 */
double cpuTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
}

int main()
{

    // 1. Inicialización de datos
    // -------------- CPU --------------
    cv::Mat h_rgbImg = loadImage();
    cv::Mat h_greyImg(h_rgbImg.size(), CV_8UC1);
    int rows = h_rgbImg.rows, cols = h_rgbImg.cols;
    size_t RGBbytes = cols * rows * sizeof(uchar3);
    size_t Greybytes = cols * rows * sizeof(uchar);


    /// 2. Asignación de memoria dinámica
    // ----------------------------------
    uchar3 *d_rgbImg;
    uchar *d_greyImg;
    cudaMalloc(&d_rgbImg, RGBbytes);
    cudaMalloc(&d_greyImg, Greybytes);
    // ----------------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    cudaMemcpy(d_rgbImg, h_rgbImg.ptr<uchar3>(), RGBbytes, cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    double tic, toc, timeCPU, timeGPU;

    // 5. Configuración de la ejecución en la GPU.
    // -------------------------------------------

    int bCols = (cols <= 32)? cols : 32, bRows = (rows <= 32)? rows : 32;
    dim3 block(bCols, bRows);

    int gCols = (cols + block.x - 1)/ block.x, gRows = (rows + block.y - 1)/ block.y;
    dim3 grid(gCols, gRows);

    printf("Grid: %d x %d\n", gCols, gRows);
    printf("Block: %d x %d\n", bCols, bRows);

    tic = cpuTime();
    imageToGreyScale<<<grid, block>>>(d_rgbImg, d_greyImg, rows, cols);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    printf("GPU time: %lf segs.\n", timeGPU);
    // -------------------------------------------



    return 0;
}