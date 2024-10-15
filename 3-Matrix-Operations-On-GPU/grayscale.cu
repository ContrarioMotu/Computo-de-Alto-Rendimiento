#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

/**
 * Carga una imagen a color.
 * 
 * @returns La imagen cargada.
 */
cv::Mat loadImage(){
    cv::Mat image = cv::imread("Images/BananaCat.jpg", cv::IMREAD_COLOR);

    if (image.empty())
    {
        std::cerr << "ERROR: No se pudo abrir o encontrar la imagen" << std::endl;
        exit(1);
    }

    return image;
}

/**
 * (Kernel) Convierte una imagen a color a escala de grises, utilizando la GPU.
 *
 * @param rgb: Matriz de color a convertir.
 * @param grey: Matriz de escala de grises.
 * @param rows: Número de filas de la matriz.
 * @param cols: Número de columnas de la matriz.
 */
__global__ void imageToGreyScale(uchar3 *rgb, uchar *grey, int rows, int cols)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows)
    {
        uchar b = rgb[((row * cols) + col)].x;
        uchar g = rgb[((row * cols) + col)].y;
        uchar r = rgb[((row * cols) + col)].z;

        grey[(row * cols) + col] = r * 0.21f + g * 0.71f + b * 0.07f;
    }
}

/**
 * Convierte una imagen a color a escala de grises, utilizando la CPU.
 *
 * @param rgb: Matriz de color a convertir.
 * @param grey: Matriz de escala de grises.
 * @param rows: Número de filas de la matriz.
 * @param cols: Número de columnas de la matriz.
 */
void imageToGreyScaleCPU(uchar3 *rgb, uchar *grey, int rows, int cols){
    uchar r, g, b;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            b = rgb[((i * cols) + j)].x;
            g = rgb[((i * cols) + j)].y;
            r = rgb[((i * cols) + j)].z;
            grey[(i * cols) + j] = r * 0.21f + g * 0.71f + b * 0.07f;
        }
    }
}

/**
 * Valida que los resultados obtenidos en la GPU y en la CPU sean iguales.
 *
 * @param h_greyImgGPU: Matriz de escala de grises obtenida en la GPU.
 * @param h_greyImgCPU: Matriz de escala de grises obtenida en la CPU.
 * @param rows: Número de filas de la matriz.
 * @param cols: Número de columnas de la matriz.
 */
void validate(uchar *h_greyImgGPU, uchar *h_greyImgCPU, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        if(h_greyImgGPU[i] - h_greyImgCPU[i] > 1){
            std::cout << "[" << i << "] : " << h_greyImgGPU[i]
                        << " != " << h_greyImgCPU[i] << "\n"
                        << "ERROR: Resultado distinto" << std::endl;
            return;
        }
    }
    std::cout << "ESCALA DE GRISES CORRECTA" << std::endl;
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

/**
 * Calcula el SpeedUp.
 *
 * @param h_time: Tiempo utilizado por el Host (CPU) para realizar la operación.
 * @param d_time: Tiempo utilizado por el Host (GPU) para realizar la operación.
 *
 * @returns El SpeedUp obtenido del algoritmo ejecutado en paralelo en comparación con su ejecución
 * en serie en una sola unidad de procesamiento.
 */
double speedUp(double h_time, double d_time)
{
    return h_time / d_time;
}

int main()
{

    // 1. Inicialización de datos
    // -------------- CPU --------------
    cv::Mat h_rgbImg = loadImage();
    cv::Mat h_greyImg(h_rgbImg.size(), CV_8UC1);
    cv::Mat h_greyImgGPU(h_rgbImg.size(), CV_8UC1);
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

    // 6. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_greyImgGPU.ptr<uchar>(), d_greyImg, Greybytes, cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    // Producto de matrices en CPU.
    // -------------------------------
    tic = cpuTime();
    imageToGreyScaleCPU(h_rgbImg.ptr<uchar3>(), h_greyImg.ptr<uchar>(), rows, cols);
    toc = cpuTime();
    timeCPU = toc - tic;
    printf("CPU time: %lf segs.\n", timeCPU);
    // -------------------------------

    // 7. Validación de resultados.
    // ------------------------------
    validate(h_greyImgGPU.ptr<uchar>(), h_greyImg.ptr<uchar>(), rows, cols);
    // ------------------------------

    cv::imwrite("Images/BananaCatGreyGPU.jpg", h_greyImgGPU);
    cv::imwrite("Images/BananaCatGreyCPU.jpg", h_greyImg);

    // 8. Liberación de memoria.
    // ---------------------------
    cudaFree(d_rgbImg);
    cudaFree(d_greyImg);
    // ---------------------------

    // 9. SpeedUp.
    // -----------
    printf("SpeedUp : %lf\n", speedUp(timeCPU, timeGPU));
    // -----------


    return 0;
}