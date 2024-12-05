#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define INPUT 784  // 28*28 pixels
#define HIDDEN 20 // Number of hidden nodes
#define OUTPUT 10  // 10 digits (0-9)

#define TRAINING_SET 60000
#define TEST_SET 10000

#define EPOCHS 20


void loadDataset(double **trainingSet, double **trainingLabels, double **testSet, double **testLabels){
    FILE *trainingSetFile = fopen("mnist_train_images.bin", "rb");
    if (trainingSetFile == NULL)
    {
        printf("Error al cargar el conjunto de entrenamiento...\n");
        exit(1);
    }

    FILE *trainingLabelsFile = fopen("mnist_train_labels.bin", "rb");
    if (trainingLabelsFile == NULL)
    {
        printf("Error al cargar las etiquetas de entrenamiento...\n");
        exit(1);
    }

    FILE *testSetFile = fopen("mnist_test_images.bin", "rb");
    if (testSetFile == NULL)
    {
        printf("Error al cargar el conjunto de prueba...\n");
        exit(1);
    }

    FILE *testLabelsFile = fopen("mnist_test_labels.bin", "rb");
    if (testLabelsFile == NULL)
    {
        printf("Error al cargar las etiquetas de prueba...\n");
        exit(1);
    }

    for (int i = 0; i < TRAINING_SET; i++)
    {
        for (int j = 0; j < INPUT; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, trainingSetFile);
            trainingSet[i][j] = (double)pixel / 255.0;
        }
    }

    for (int i = 0; i < TRAINING_SET; i++){
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, trainingLabelsFile);
        for (int j = 0; j < OUTPUT; j++){
            trainingLabels[i][j] = (j == label)? 1 : 0;
        }
    }

    for (int i = 0; i < TEST_SET; i++)
    {
        for (int j = 0; j < INPUT; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, testSetFile);
            testSet[i][j] = (double)pixel / 255.0;
        }
    }

    for (int i = 0; i < TEST_SET; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, testLabelsFile);
        for (int j = 0; j < OUTPUT; j++)
        {
            testLabels[i][j] = (j == label)? 1 : 0;
        }
    }

    fclose(trainingSetFile);
    fclose(trainingLabelsFile);
    fclose(testSetFile);
    fclose(testLabelsFile);
}

__device__ double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}



double crossEntropy(double **labels, double **predicts, int size, int clases) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < clases; j++) {
            sum += labels[i][j] * log(1.0E-15 + predicts[i][j]);
        }
    }
    return -sum / size;
}

__device__ int max_index(double *out, int size) {
    int max_i = 0;
    for (int i = 0; i < size; i++) {
        printf("out[%d] = %f\n", i, out[i]);
        if (out[i] > out[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

double *vectorize(double **matrix, int rows, int columns) {
    double *vector = (double*)malloc(rows * columns * sizeof(double));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            vector[(columns * i) + j] = matrix[i][j];
        }
    }
    return vector;
}

void printVector(double *A, int size){
    printf("[");
    for(int i=0; i < size; i++){
        printf("%.2f ", A[i]);
    }
    printf("]\n");
}

__global__ void confusionMatrix(double *labels, double *predicts, int *matrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l, p;
    printf("Conf matrix done...\n");
    if (idx < size) {
        l = max_index(labels + (idx * OUTPUT), OUTPUT);
        printf("    • Label: %d\n", l);
        p = max_index(predicts + (idx * OUTPUT), OUTPUT);
        printf("    • Predicted: %d\n", p);
        atomicAdd(&matrix[(p * OUTPUT) + l], 1);
    }
    __syncthreads();
}

// double *backPropagate(double *input, double *output, double **W1, double **W2, double *b1, double *b2){
//     double error[OUTPUT];
//     double hidden[HIDDEN];
//     double *prediction = (double*)malloc(OUTPUT * sizeof(double));

//     for (int i = 0; i < HIDDEN; i++)
//     {
//         double sum = 0;
//         for (int j = 0; j < INPUT; j++)
//         {
//             sum += input[j] * W1[j][i];
//         }
//         sum += b1[i];
//         hidden[i] = sigmoid(sum);
//     }
//     for (int i = 0; i < OUTPUT; i++)
//     {
//         double sum = 0;
//         for (int j = 0; j < HIDDEN; j++)
//         {
//             sum += hidden[j] * W2[j][i];
//         }
//         sum += b2[i];
//         prediction[i] = sigmoid(sum);
//     }

//     for (int i = 0; i < OUTPUT; i++)
//     {
//         error[i] = output[i] - prediction[i];
//     }
//     double delta2[HIDDEN];
//     for (int i = 0; i < HIDDEN; i++)
//     {
//         delta2[i] = 0;
//         for (int j = 0; j < OUTPUT; j++)
//         {
//             delta2[i] += error[j] * prediction[j] * (1 - prediction[j]) * W2[i][j];
//         }
//     }
//     double delta1[INPUT];
//     for (int i = 0; i < INPUT; i++)
//     {
//         delta1[i] = 0;
//         for (int j = 0; j < HIDDEN; j++)
//         {
//             delta1[i] += delta2[j] * hidden[j] * (1 - hidden[j]) * W1[i][j];
//         }
//     }

//     double learning_rate = 0.01;
//     for (int i = 0; i < INPUT; i++)
//     {
//         for (int j = 0; j < HIDDEN; j++)
//         {
//             W1[i][j] += learning_rate * delta1[i] * input[j];
//         }
//     }
//     for (int i = 0; i < HIDDEN; i++)
//     {
//         b1[i] += learning_rate * delta2[i];
//         for (int j = 0; j < OUTPUT; j++)
//         {
//             W2[i][j] += learning_rate * error[j] * prediction[j] * (1 - prediction[j]) * hidden[i];
//         }
//     }
//     for (int i = 0; i < OUTPUT; i++)
//     {
//         b2[i] += learning_rate * error[i] * prediction[i] * (1 - prediction[i]);
//     }

//     return prediction;
// }

__global__ void feedForwardOnGPU(double *in, double *pred, double *W1, double *W2, double *b1, double *b2){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double hidden[HIDDEN];
    if(idx < HIDDEN){
        double sum = 0.0;
        for (int i = 0; i < INPUT; i++)
        {
            sum += in[i] * W1[(i * HIDDEN) + idx];
            
        }
        sum += b1[idx];
        hidden[idx] = sigmoid(sum);
        __syncthreads();
    }
    if(idx < OUTPUT){
        double sum = 0.0;
        for (int i = 0; i < HIDDEN; i++)
        {
            sum += hidden[i] * W2[(i * OUTPUT) + idx];
        }
        sum += b2[idx];
        pred[idx] = sigmoid(sum);
        __syncthreads();
    }
}

void saveModel(const char *file_name, double **W1, double *b1, double **W2, double *b2) {
    FILE* file = fopen(file_name, "wb");
    if (file == NULL) {
        printf("Error al guardar el modelo...\n");
        exit(1);
    }
    fwrite(W1, sizeof(double), HIDDEN * INPUT, file);
    fwrite(W2, sizeof(double), HIDDEN * OUTPUT, file);
    fwrite(b1, sizeof(double), HIDDEN, file);
    fwrite(b2, sizeof(double), OUTPUT, file);
    fclose(file);
}

/*
 * Función que obtiene la hora en segundos.
 *
 * @returns El tiempo actual del cpu en segundos.
 */
double cpuTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.0E-6);
}

// void train(double **input, double **output, double **W1, double **W2, double *b1, double *b2)
// {
//     double tic, toc, time;
//     double **predicts = (double**)malloc(TEST_SET * sizeof(double*));
//     for(int e = 0; e < EPOCHS; e++){
//         tic = cpuTime();
//         for (int i = 0; i < TEST_SET; i++)
//         {
//             predicts[i] = (double*)malloc(OUTPUT * sizeof(double));
//             predicts[i] = backPropagate(input[i], output[i], W1, W2, b1, b2);
//         }
//         toc = cpuTime();
//         time = toc - tic;
//         double loss = crossEntropy(output, predicts, TEST_SET, OUTPUT);
//         printf("Epoch %d -> Elapsed Time: %lf, Loss: %f\n", e, time, loss);
//     }
//     saveModel("model.bin", W1, b1, W2, b2);
//     free(predicts);
// }

void printConfusionMatrix(int *matrix){

    char *sep = (char*)malloc(sizeof(char) * 7 * OUTPUT);
    for (int i = 0; i < 7*OUTPUT; i++)
    {
        sep[i] = '-';
    }

    printf("\nConfusion Matrix:\n");
    printf("      ");
    for (int i = 0; i < OUTPUT; i++){
        printf("   %d   ", i);
    }
    printf("\n      %s\n", sep);

    for (int i = 0; i < OUTPUT; i++){
        printf("%-6d|", i * OUTPUT);
        for (int j = 0; j < OUTPUT; j++){
            printf("%-6d|", matrix[(i * OUTPUT) + j]);
        }
        printf("\n      %s\n", sep);
    }

    free(sep);
}

// void test(double **input, double **output, double **W1, double **W2, double *b1, double *b2){
//     double **predicts = (double**)malloc(sizeof(double*) * TEST_SET);
//     double tic, toc, time;

//     tic = cpuTime();
//     for (int i = 0; i < TEST_SET; i++)
//     {
//         predicts[i] = (double*)malloc(sizeof(double) * OUTPUT);
//         predicts[i] = feedForward(input[i], W1, W2, b1, b2);
//     }

//     toc = cpuTime();
//     time = toc - tic;
//     int **matrix = confusionMatrix(output, predicts, TEST_SET);
//     int correct_predictions = 0;
//     for (int i = 0; i < OUTPUT; i++)
//     {
//         correct_predictions += matrix[i][i];
//     }
//     printConfusionMatrix(matrix);
//     printf("Testing on CPU -> Accuracy: %.6f, Elapsed time: %.6f\n",
//                         (double) correct_predictions / TEST_SET, time);
//     free(predicts);
// }

void test(double *input, double *output, double *W1, double *W2, double *b1, double *b2){
    double *predicts; cudaMalloc((double **)&predicts, TEST_SET * OUTPUT * sizeof(double));
    double tic, toc, time;
    dim3 threads(32);
    dim3 blocks((INPUT > OUTPUT)? (INPUT + threads.x - 1) / threads.x : (OUTPUT + threads.x - 1) / threads.x);
    printf("Threads: %d\n", threads.x);
    printf("Blocks: %d\n", blocks.x);
    tic = cpuTime();
    for (int i = 0; i < TEST_SET; i++)
    {
        feedForwardOnGPU<<<blocks, threads>>>(input + (i*TEST_SET), predicts + (i*TEST_SET), W1, W2, b1, b2);
    }
    toc = cpuTime();
    time = toc - tic;
    printf("Testing on GPU -> Elapsed time: %.6f\n", time);

    int *matrix; cudaMalloc((void **)&matrix, OUTPUT * OUTPUT * sizeof(int));
    blocks = dim3((TEST_SET + threads.x - 1) / threads.x);
    printf("Threads: %d\n", threads.x);
    printf("Blocks: %d\n", blocks.x);
    confusionMatrix<<<blocks, threads>>>(output, predicts, matrix, TEST_SET);
    int correct_predictions = 0;
    for (int i = 0; i < OUTPUT; i++)
    {
        correct_predictions += matrix[(i * OUTPUT) + i];
    }
    printf("Testing on GPU -> Accuracy: %.6f, Elapsed time: %.6f\n",
                        (double) correct_predictions / TEST_SET, time);
    printConfusionMatrix(matrix);
    free(predicts);
}

void loadModel(char* file_name, double **W1, double *b1, double **W2, double *b2) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("Error al cargar el modelo...\n");
        exit(1);
    }
    fread(W1, sizeof(double), HIDDEN * INPUT, file);
    fread(W2, sizeof(double), HIDDEN * OUTPUT, file);
    fread(b1, sizeof(double), HIDDEN, file);
    fread(b2, sizeof(double), OUTPUT, file);
    fclose(file);
}

void genRandWeights(double **W1, double *b1, double **W2, double *b2) {
    for (int i = 0; i < INPUT; i++)
    {
        for (int j = 0; j < HIDDEN; j++)
        {
            W1[i][j] = (double) ((rand() % 2000001)*0.000001) - 1.0;
        }
    }

    for (int i = 0; i < HIDDEN; i++)
    {
        b1[i] = (double) ((rand() % 2000001)*0.000001) - 1.0;
        for (int j = 0; j < OUTPUT; j++)
        {
            W2[i][j] = (double) ((rand() % 2000001)*0.000001) - 1.0;
        }
    }

    for (int i = 0; i < OUTPUT; i++)
    {
        b2[i] = (double) ((rand() % 2000001)*0.000001) - 1.0;
    }
}

int main()
{
    double **X_train, **Y_train, **X_test, **Y_test, **W1, **W2, *b1, *b2;
    double *d_X_train, *d_Y_train, *d_X_test, *d_Y_test, *d_W1, *d_W2, *d_b1, *d_b2;

    X_train = (double**)malloc(TRAINING_SET * sizeof(double*));
    Y_train = (double**)malloc(TRAINING_SET * sizeof(double*));
    for (int i = 0; i < TRAINING_SET; i++)
    {
        X_train[i] = (double*)malloc(INPUT * sizeof(double));
        Y_train[i] = (double*)malloc(OUTPUT * sizeof(double));
    }

    
    X_test = (double**)malloc(TEST_SET * sizeof(double*));
    Y_test = (double**)malloc(TEST_SET * sizeof(double*));
    for (int i = 0; i < TEST_SET; i++)
    {
        X_test[i] = (double*)malloc(INPUT * sizeof(double));
        Y_test[i] = (double*)malloc(OUTPUT * sizeof(double));
    }
    W1 = (double**)malloc(INPUT * sizeof(double*));
    for (int i = 0; i < INPUT; i++)
    {
        W1[i] = (double*)malloc(HIDDEN * sizeof(double));
    }
    W2 = (double**)malloc(HIDDEN * sizeof(double*));
    for (int i = 0; i < HIDDEN; i++)
    {
        W2[i] = (double*)malloc(OUTPUT * sizeof(double));
    }
    b1 = (double*)malloc(HIDDEN * sizeof(double));
    b2 = (double*)malloc(OUTPUT * sizeof(double));

    cudaMalloc((void**)&d_X_train, TRAINING_SET * INPUT * sizeof(double));
    cudaMalloc((void**)&d_Y_train, TRAINING_SET * OUTPUT * sizeof(double));
    cudaMalloc((void**)&d_X_test, TEST_SET * INPUT * sizeof(double));
    cudaMalloc((void**)&d_Y_test, TEST_SET * OUTPUT * sizeof(double));
    cudaMalloc((void**)&d_W1, HIDDEN * INPUT * sizeof(double));
    cudaMalloc((void**)&d_W2, HIDDEN * OUTPUT * sizeof(double));
    cudaMalloc((void**)&d_b1, HIDDEN * sizeof(double));
    cudaMalloc((void**)&d_b2, OUTPUT * sizeof(double));

    printf("• Network initialized...\n");

    loadDataset(X_train, Y_train, X_test, Y_test);

    double *h_X_train = vectorize(X_train, TRAINING_SET, INPUT);
    double *h_Y_train = vectorize(Y_train, TRAINING_SET, OUTPUT);
    double *h_X_test = vectorize(X_test, TEST_SET, INPUT);
    double *h_Y_test = vectorize(Y_test, TEST_SET, OUTPUT);

    cudaMemcpy(d_X_train, h_X_train, TRAINING_SET * INPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_train, h_Y_train, TRAINING_SET * OUTPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X_test, h_X_test, TEST_SET * INPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_test, h_Y_test, TEST_SET * OUTPUT * sizeof(double), cudaMemcpyHostToDevice);

    printf("• Dataset loaded...\n");

    genRandWeights(W1, b1, W2, b2);

    double *h_W1 = vectorize(W1, INPUT, HIDDEN);
    double *h_W2 = vectorize(W2, HIDDEN, OUTPUT);

    cudaMemcpy(d_W1, h_W1, HIDDEN * INPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, HIDDEN * OUTPUT * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, HIDDEN * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, OUTPUT * sizeof(double), cudaMemcpyHostToDevice);

    printf("• Weights initialized...\n");

    printVector(d_X_test, TEST_SET * INPUT);

    //train(h_X_train, h_Y_train, h_W1, h_W2, h_b1, h_b2);
    //test(d_X_test, d_Y_test, d_W1, d_W2, d_b1, d_b2);

    free(h_X_train);
    free(h_Y_train);
    free(h_X_test);
    free(h_Y_test);
    free(h_W1);
    free(h_W2);
    free(b1);
    free(b2);

    cudaFree(d_X_train);
    cudaFree(d_Y_train);
    cudaFree(d_X_test);
    cudaFree(d_Y_test);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_b1);
    cudaFree(d_b2);

    return 0;
}