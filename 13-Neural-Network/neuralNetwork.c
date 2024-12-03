#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>

#define INPUT 784  // 28*28 pixels
#define HIDDEN 256 // Number of hidden nodes
#define OUTPUT 10  // 10 digits (0-9)

#define TRAINING_SET 60000
#define TEST_SET 10000

#define EPOCHS 2

void initializeNetwork(double **X_train, double **Y_train, double **X_test, double **Y_test,
                        double **W1, double **W2, double *b1, double *b2){
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
}

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

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// __device__ double sigmoid(double x){
//     return 1.0 / (1.0 + exp(-x));
// }

double crossEntropy(double **labels, double **predicts, int size, int clases) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < clases; j++) {
            sum += labels[i][j] * log(1.0E-15 + predicts[i][j]);
        }
    }
    return -sum / size;
}

int max_index(double *out, int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
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

int **confusionMatrix(double **labels, double **predicts, int size) {
    int **matrix = (int**)malloc(OUTPUT * sizeof(int*));
    for (int i = 0; i < OUTPUT; i++) {
        matrix[i] = (int*)malloc(OUTPUT * sizeof(int));
    }

    int l, p;
    for (int i = 0; i < size; i++)
    {
        l = max_index(labels[i], OUTPUT);
        p = max_index(predicts[i], OUTPUT);
        matrix[p][l]++;
    }

    return matrix;
}

double *backPropagate(double *input, double *output, double **W1, double **W2, double *b1, double *b2){
    double error[OUTPUT];
    double hidden[HIDDEN];
    double *prediction = (double*)malloc(OUTPUT * sizeof(double));

    for (int i = 0; i < HIDDEN; i++)
    {
        double sum = 0;
        for (int j = 0; j < INPUT; j++)
        {
            sum += input[j] * W1[j][i];
        }
        sum += b1[i];
        hidden[i] = sigmoid(sum);
    }
    for (int i = 0; i < OUTPUT; i++)
    {
        double sum = 0;
        for (int j = 0; j < HIDDEN; j++)
        {
            sum += hidden[j] * W2[j][i];
        }
        sum += b2[i];
        prediction[i] = sigmoid(sum);
    }

    for (int i = 0; i < OUTPUT; i++)
    {
        error[i] = output[i] - prediction[i];
    }
    double delta2[HIDDEN];
    for (int i = 0; i < HIDDEN; i++)
    {
        delta2[i] = 0;
        for (int j = 0; j < OUTPUT; j++)
        {
            delta2[i] += error[j] * prediction[j] * (1 - prediction[j]) * W2[i][j];
        }
    }
    double delta1[INPUT];
    for (int i = 0; i < INPUT; i++)
    {
        delta1[i] = 0;
        for (int j = 0; j < HIDDEN; j++)
        {
            delta1[i] += delta2[j] * hidden[j] * (1 - hidden[j]) * W1[i][j];
        }
    }

    double learning_rate = 0.1;
    for (int i = 0; i < INPUT; i++)
    {
        for (int j = 0; j < HIDDEN; j++)
        {
            W1[i][j] += learning_rate * delta1[i] * input[j];
        }
    }
    for (int i = 0; i < HIDDEN; i++)
    {
        b1[i] += learning_rate * delta2[i];
        for (int j = 0; j < OUTPUT; j++)
        {
            W2[i][j] += learning_rate * error[j] * prediction[j] * (1 - prediction[j]) * hidden[i];
        }
    }
    for (int i = 0; i < OUTPUT; i++)
    {
        b2[i] += learning_rate * error[i] * prediction[i] * (1 - prediction[i]);
    }

    return prediction;
}

double *feedForward(double *input, double **W1, double **W2, double *b1, double *b2){
    double hidden[HIDDEN] = {0.0};
    double *prediction = (double*)malloc(OUTPUT * sizeof(double));

    for (int i = 0; i < HIDDEN; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < INPUT; j++)
        {
            sum += input[j] * W1[j][i];
        }
        sum += b1[i];
        hidden[i] = sigmoid(sum);
    }
    for (int i = 0; i < OUTPUT; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < HIDDEN; j++)
        {
            sum += hidden[j] * W2[j][i];
        }
        sum += b2[i];
        prediction[i] = sigmoid(sum);
    }
    return prediction;
}

// __global__ void feedForwardOnGPU(double *input, double *pred, double *W1, double *W2, double *b1, double *b2){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     __shared__ double hidden[HIDDEN];
//     if(idx < HIDDEN){
//         double sum = 0.0;
//         for (int i = 0; i < INPUT; i++)
//         {
//             sum += input[i] * W1[(i * HIDDEN) + idx];
//         }
//         sum += b1[idx];
//         hidden[idx] = sigmoid(sum);
//         __syncthreads();
//     }

//     if(idx < OUTPUT){
//         double sum = 0.0;
//         for (int i = 0; i < HIDDEN; i++)
//         {
//             sum += hidden[i] * W2[(i * OUTPUT) + idx];
//         }
//         sum += b2[idx];
//         pred[idx] = sigmoid(sum);
//         __syncthreads();
//     }
// }

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

void train(double **input, double **output, double **W1, double **W2, double *b1, double *b2)
{
    double tic, toc, time;
    double **predicts = (double**)malloc(TEST_SET * sizeof(double*));
    for(int e = 0; e < EPOCHS; e++){
        tic = cpuTime();
        for (int i = 0; i < TEST_SET; i++)
        {
            predicts[i] = (double*)malloc(OUTPUT * sizeof(double));
            predicts[i] = backPropagate(input[i], output[i], W1, W2, b1, b2);
        }
        toc = cpuTime();
        time = toc - tic;
        double loss = crossEntropy(output, predicts, TEST_SET, OUTPUT);
        printf("Epoch %d -> Elapsed Time: %lf, Loss: %f\n", e, time, loss);
    }
    saveModel("model.bin", W1, b1, W2, b2);
    free(predicts);
}

void printConfusionMatrix(int **matrix){

    char *sep = (char*)mallor(sizeof(char) * 7 * OUTPUT);
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
        printf("%-6d|", i);
        for (int j = 0; j < OUTPUT; j++){
            printf("%-6d|", matrix[i][j]);
        }
        printf("\n      %s\n", sep);
    }

    free(sep);
}

void test(double **input, double **output, double **W1, double **W2, double *b1, double *b2){
    double **predicts = (double**)malloc(sizeof(double*) * TEST_SET);
    double tic, toc, time;

    tic = cpuTime();
    for (int i = 0; i < TEST_SET; i++)
    {
        predicts[i] = (double*)malloc(sizeof(double) * OUTPUT);
        predicts[i] = feedForward(input[i], W1, W2, b1, b2);
    }

    toc = cpuTime();
    time = toc - tic;
    int **matrix = confusionMatrix(output, predicts, TEST_SET);
    int correct_predictions = 0;
    for (int i = 0; i < OUTPUT; i++)
    {
        correct_predictions += matrix[i][i];
    }
    printConfusionMatrix(matrix);
    printf("Testing on CPU -> Accuracy: %.6f, Elapsed time: %.6f\n",
                        (double) correct_predictions / TEST_SET, time);
    free(predicts);
}

// void testOnGPU(double *input, double *output, double *W1, double *W2, double *b1, double *b2){
//     double *predicts = new double[TEST_SET * OUTPUT];
//     double tic, toc, time;


//     tic = cpuTime();
//     for (int i = 0; i < TEST_SET; i++)
//     {
//         feedForwardOnGPU(input + (i*TEST_SET), predicts + (i*TEST_SET), W1, W2, b1, b2);
//     }

//     toc = cpuTime();
//     time = toc - tic;
//     int **matrix = confusionMatrix(output, predicts, TEST_SET, OUTPUT);
//     int correct_predictions = 0;
//     for (int i = 0; i < OUTPUT; i++)
//     {
//         correct_predictions += matrix[i][i];
//     }
//     printConfusionMatrix(matrix);
//     printf("Testing on GPU -> Accuracy: %.6f, Elapsed time: %.6f\n",
//                         (double) correct_predictions / TEST_SET, time);
//     free(predicts);
// }

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
            W1[i][j] = (double) ((rand() % 20001)*0.0001) - 1.0;
        }
    }

    for (int i = 0; i < HIDDEN; i++)
    {
        b1[i] = (double) ((rand() % 20001)*0.0001) - 1.0;
        for (int j = 0; j < OUTPUT; j++)
        {
            W2[i][j] = (double) ((rand() % 20001)*0.0001) - 1.0;
        }
    }

    for (int i = 0; i < OUTPUT; i++)
    {
        b2[i] = (double) ((rand() % 20001)*0.0001) - 1.0;
    }
}

int main()
{
    double **h_X_train, **h_Y_train, **h_X_test, **h_Y_test, **h_W1, **h_W2, *h_b1, *h_b2;
    //double *d_X_train, *d_Y_train, *d_X_test, *d_Y_test, *d_W1, *d_W2, *d_b1, *d_b2;

    initializeNetwork(h_X_train, h_Y_train, h_X_test, h_Y_test, h_W1, h_W2, h_b1, h_b2);

    loadDataset(h_X_train, h_Y_train, h_X_test, h_Y_test);

    printf("• Dataset loaded...\n");

    genRandWeights(h_W1, h_b1, h_W2, h_b2);

    printf("• Weights initialized...\n");

    train(h_X_train, h_Y_train, h_W1, h_W2, h_b1, h_b2);
    test(h_X_test, h_Y_test, h_W1, h_W2, h_b1, h_b2);

    free(h_X_train);
    free(h_Y_train);
    free(h_X_test);
    free(h_Y_test);
    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);

    return 0;
}