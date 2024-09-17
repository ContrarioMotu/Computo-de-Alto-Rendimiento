#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BLUR_SIZE 3

/**
 * Carga una imagen a color.
 * 
 * @returns La imagen cargada.
 */
Mat loadImage(){
    Mat image = imread("Imagenes/imgPrueba.jpg", IMREAD_COLOR);

    if (image.empty())
    {
        cerr << "ERROR: No se pudo abrir o encontrar la imagen" << endl;
        exit(1);
    }

    return image;
}

/**
 * (Kernel) Agrega un filtro de desenfoque a una imagen, utilizando la GPU.
 *
 * @param image: Vector de la imagen original.
 * @param blur: Vector de la imagen desenfocada.
 * @param rows: Número de filas de la imagen.
 * @param cols: Número de columnas de la imagen.
 */
__global__ void blurImageGPU(uchar *image, uchar *blur, int rows, int cols)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows)
    {
        int pixVal = 0, pixels = 0, curRow, curCol;

        for(int i = -BLUR_SIZE; i < BLUR_SIZE + 1; ++i){
            for(int j = -BLUR_SIZE; j < BLUR_SIZE + 1; ++j){
                curRow = row + i;
                curCol = col + j;
                if (curRow > -1 && curRow < rows && curCol > -1 && curCol < cols)
                {
                    pixVal += image[(curRow * cols) + curCol];
                    pixels++;
                }
            }
        }
        blur[(row * cols) + col] = (uchar)(pixVal / pixels);
    }
}

/**
 * Agrega un filtro de desenfoque a una imagen, utilizando la CPU.
 *
 * @param image: Vector de la imagen original.
 * @param blur: Vector de la imagen desenfocada.
 * @param rows: Número de filas de la imagen.
 * @param cols: Número de columnas de la imagen.
 */
void blurImageCPU(uchar *image, uchar *blur, int rows, int cols){
    for(int row = 0; row < rows; ++row){
        for(int col = 0; col < cols; ++col){
            int pixVal = 0, pixels = 0, curRow, curCol;
            for(int i = -BLUR_SIZE; i < BLUR_SIZE + 1; ++i){
                for(int j = -BLUR_SIZE; j < BLUR_SIZE + 1; ++j){
                    curRow = row + i;
                    curCol = col + j;
                    if (curRow > -1 && curRow < rows && curCol > -1 && curCol < cols)
                    {
                        pixVal += image[(curRow * cols) + curCol];
                        pixels++;
                    }
                }
            }

        blur[(row * cols) + col] = (uchar)(pixVal / pixels);
        }

    }
}

/**
 * Valida que los resultados obtenidos en la GPU y en la CPU sean iguales.
 *
 * @param h_bluredImgGPU: Vector del filtro de desenfoque, obtenido en la GPU.
 * @param h_bluredImgCPU: Vector del filtro de desenfoque, obtenido en la CPU.
 * @param rows: Número de filas de la imagen.
 * @param cols: Número de columnas de la imagen.
 */
void validate(uchar *h_bluredImgGPU, uchar *h_bluredImgCPU, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        if(h_bluredImgGPU[i] - h_bluredImgCPU[i] > 1){
            cout << "[" << i << "] : " << (int)h_bluredImgGPU[i]
                        << " != " << (int)h_bluredImgCPU[i] << "\n"
                        << "ERROR: Resultado distinto\n";
            return;
        }
    }
    cout << "FILTRO CORRECTO\n";
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
    Mat h_Img = loadImage();
    int rows = h_Img.rows, cols = h_Img.cols;
    int channelBytes = cols * rows * sizeof(uchar);
    vector<Mat> channelsGPU(3), channelsCPU(3);
    split(h_Img, channelsGPU);
    split(h_Img, channelsCPU);

    /// 2. Asignación de memoria dinámica
    // ----------------------------------
    uchar *h_bluredImgR, *h_bluredImgG, *h_bluredImgB, *d_ImgR, *d_ImgG, *d_ImgB;    
    uchar *d_bluredImgR, *d_bluredImgG, *d_bluredImgB;

    h_bluredImgR = new uchar[channelBytes];
    h_bluredImgG = new uchar[channelBytes];
    h_bluredImgB = new uchar[channelBytes];

    cudaMalloc((void**)&d_ImgR, channelBytes);
    cudaMalloc((void**)&d_ImgG, channelBytes);
    cudaMalloc((void**)&d_ImgB, channelBytes);

    cudaMalloc((void**)&d_bluredImgR, channelBytes);
    cudaMalloc((void**)&d_bluredImgG, channelBytes);
    cudaMalloc((void**)&d_bluredImgB, channelBytes);
    // ----------------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    cudaMemcpy(d_ImgR, channelsGPU[2].ptr(), channelBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ImgG, channelsGPU[1].ptr(), channelBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ImgB, channelsGPU[0].ptr(), channelBytes, cudaMemcpyHostToDevice);
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
    blurImageGPU<<<grid, block>>>(d_ImgR, d_bluredImgR, rows, cols);
    blurImageGPU<<<grid, block>>>(d_ImgG, d_bluredImgG, rows, cols);
    blurImageGPU<<<grid, block>>>(d_ImgB, d_bluredImgB, rows, cols);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    cout << "GPU time: " << timeGPU << " segs." << endl;
    // -------------------------------------------

    // 6. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(channelsGPU[2].ptr(), d_bluredImgR, channelBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(channelsGPU[1].ptr(), d_bluredImgG, channelBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(channelsGPU[0].ptr(), d_bluredImgB, channelBytes, cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    // Producto de matrices en CPU.
    // -------------------------------
    tic = cpuTime();
    blurImageCPU(channelsCPU[2].ptr(), h_bluredImgR, rows, cols);
    blurImageCPU(channelsCPU[1].ptr(), h_bluredImgG, rows, cols);
    blurImageCPU(channelsCPU[0].ptr(), h_bluredImgB, rows, cols);
    toc = cpuTime();
    timeCPU = toc - tic;
    cout << "CPU time: " << timeCPU << " segs." << endl;
    // -------------------------------

    // 7. Validación de resultados.
    // ------------------------------
    cout << "CANAL ROJO: ... ";
    validate(channelsGPU[2].ptr(), h_bluredImgR, rows, cols);
    cout << "CANAL VERDE: ... ";
    validate(channelsGPU[1].ptr(), h_bluredImgG, rows, cols);
    cout << "CANAL AZUL: ... ";
    validate(channelsGPU[0].ptr(), h_bluredImgB, rows, cols);
    // ------------------------------

    Mat bluredImage;
    merge(channelsGPU, bluredImage);

    imwrite("Imagenes/imgBlured.jpg", bluredImage);

    // 8. Liberación de memoria.
    // ---------------------------
    free(h_bluredImgR);
    free(h_bluredImgG);
    free(h_bluredImgB);

    cudaFree(d_ImgR);
    cudaFree(d_ImgG);
    cudaFree(d_ImgB);
    cudaFree(d_bluredImgR);
    cudaFree(d_bluredImgG);
    cudaFree(d_bluredImgB);
    // ---------------------------

    // 9. SpeedUp.
    // -----------
    cout << "SpeedUp : " << speedUp(timeCPU, timeGPU) << "\n" << endl;
    // -----------


    return 0;
}