#include <stdio.h>
#include <math.h>
#include "randlib.h"
#include "mnist/mnist.h"

#define numInputNodes 785
#define numHidden1Nodes 100
#define numHidden2Nodes 50
#define numOutputNodes 2
#define numEpochs 10

void getFloatArrayFromIntArray(float float_array[], int array[], int array_length);
void createRandomWeights(float weightArray[], int rows, int cols);
void initializeTarget(float target[], int numberOnPicture);
void getOutput(float output[], float input[], float weightArray[], int array_rows, int array_cols);
void squashOutput(float output[], int array_length);
void getOutputError(float error[], float target[], float output[], int array_length);
void getHiddenError(float error[], float hiddenLayer[], float prev_error[], float hiddenWeights[], int firstLayerLength, int secondLayerLength);
float getAverageErrorFromTarget(float target[], float outputLayer[], int array_length);
void updateWeights(float weightArray[], int array_rows, int array_cols, float layer[], float error[], float learningRate);
int testOuput(float outputLayer[], int numberOnPicture, int array_length);


int main(int argc, char const *argv[]) {
    // --- an example for working with random numbers
    seed_randoms();
    // float sampNoise = rand_frac()/2.0; // sets default sampNoise
    float sampNoise = 0;

    // --- a simple example of how to set params from the command line
    if(argc == 2) { // if an argument is provided, it is SampleNoise
            sampNoise = atof(argv[1]);
            if (sampNoise < 0 || sampNoise > .5) {

                    printf("Error: sample noise should be between 0.0 and 0.5\n");
                    return 0;
            }
    }

    // dataset for training
    mnist_data *zData;      // each image is 28x28 pixels
    unsigned int sizeData;  // depends on loadType
    int loadType = 0; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }

    // dataset for testing
    mnist_data *zData1;      // each image is 28x28 pixels
    unsigned int sizeData1;  // depends on loadType
    int loadType1 = 1; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData1, &sizeData1, loadType1)){
        printf("something went wrong loading data set\n");
        return -1;
    }

    float learningRate = 0.01;

    // creating variable to store input values
    float inputVec[numInputNodes];

    float hiddenLayer1[numHidden1Nodes];
    float hiddenLayer2[numHidden2Nodes];

    // create array to store outputs
    float outputLayer[numOutputNodes];

    float target[numOutputNodes];

    float inputLayer[numInputNodes];

    float hiddenLayer1Error[numHidden1Nodes];
    float hiddenLayer2Error[numHidden2Nodes];

    float outputLayerError[numOutputNodes];

    float analysis_output[numEpochs][2];

    // 0 = 1
    // 1 = 3
    // 2 = 0
    // 3 = 4
    // 4 = 6
    // ...

    float inputLayerWeights[numHidden1Nodes * numInputNodes];
    createRandomWeights(inputLayerWeights, numHidden1Nodes, numInputNodes);

    float hiddenLayer1Weights[numHidden2Nodes * numHidden1Nodes];
    createRandomWeights(hiddenLayer1Weights, numHidden2Nodes, numHidden1Nodes);

    float hiddenLayer2Weights[numOutputNodes * numHidden2Nodes];
    createRandomWeights(hiddenLayer2Weights, numOutputNodes, numHidden2Nodes);

    // loadType = 0, 60k training images
    // loadType = 1, 10k testing images
    // loadType = 2, 10 specific images from training set
    printf("average_error_from_target\n");

    for(int epoch = 0; epoch < numEpochs; epoch++) {

        float average_error_from_target = 0;
        // float test_output = 0;

        // testing
        for(int testImage = 0; testImage < sizeData1; testImage++) {

            // start of feed-forwarding

            get_input(inputVec, zData1, testImage, sampNoise);
            // draw_input(inputVec, zData[testImage].label);

            initializeTarget(target, inputVec, zData1[testImage].label);

            // printf("number on the picture = %d\n", zData1[testImage].label);
            // function for getting output needs float arrays,
            // so we need to convert the type of our inputVec to float
            // getFloatArrayFromIntArray(inputLayer, inputVec, numInputNodes);

            getOutput(hiddenLayer1, inputVec, inputLayerWeights, numHidden1Nodes, numInputNodes);
            squashOutput(hiddenLayer1, numHidden1Nodes);
            hiddenLayer1[numHidden1Nodes - 1] = 1; // let the last node of the hidden layer to be a bias and set it to 1

            getOutput(hiddenLayer2, hiddenLayer1, hiddenLayer1Weights, numHidden2Nodes, numHidden1Nodes);
            squashOutput(hiddenLayer2, numHidden2Nodes);
            hiddenLayer2[numHidden2Nodes - 1] = 1; // let the last node of the hidden layer to be a bias and set it to 1

            // calculate nodes for output layer and load it to outputLayer array
            getOutput(outputLayer, hiddenLayer2, hiddenLayer2Weights, numOutputNodes, numHidden2Nodes);
            squashOutput(outputLayer, numOutputNodes);
            // printf("number on the picture: %d\n", zData1[testImage].label);
            // for (int i = 0; i < numOutputNodes; i++) {
            //     printf("output[%d] = %f\n", i, outputLayer[i]);
            // }

            average_error_from_target += getAverageErrorFromTarget(target, outputLayer, numOutputNodes);
            // test_output += testOuput(outputLayer, zData1[testImage].label, numOutputNodes);

        }

        // printf("average_error_from_target[%d] = %f\n", epoch, average_error_from_target/sizeData1);

        printf("%f\n", average_error_from_target/sizeData1);


        // training
        for(int pictureIndex = 0; pictureIndex < sizeData; pictureIndex++) {

            // start of feed-forwarding

            get_input(inputVec, zData, pictureIndex, sampNoise);
            // draw_input(inputVec, zData[pictureIndex].label);

            initializeTarget(target, inputVec, zData[pictureIndex].label);

            // getFloatArrayFromIntArray(inputLayer, inputVec, numInputNodes);

            getOutput(hiddenLayer1, inputVec, inputLayerWeights, numHidden1Nodes, numInputNodes);
            squashOutput(hiddenLayer1, numHidden1Nodes);
            hiddenLayer1[numHidden1Nodes - 1] = 1; // let the last node of the hidden layer to be a bias and set it to 1

            getOutput(hiddenLayer2, hiddenLayer1, hiddenLayer1Weights, numHidden2Nodes, numHidden1Nodes);
            squashOutput(hiddenLayer2, numHidden2Nodes);
            hiddenLayer2[numHidden2Nodes - 1] = 1; // let the last node of the hidden layer to be a bias and set it to 1

            // calculate nodes for output layer and load it to outputLayer array
            getOutput(outputLayer, hiddenLayer2, hiddenLayer2Weights, numOutputNodes, numHidden2Nodes);
            squashOutput(outputLayer, numOutputNodes);

            // start of back-propogation

            getOutputError(outputLayerError, target, outputLayer, numOutputNodes);
            updateWeights(hiddenLayer2Weights, numOutputNodes, numHidden2Nodes, hiddenLayer2, outputLayerError, learningRate);

            getHiddenError(hiddenLayer2Error, hiddenLayer2, outputLayerError, hiddenLayer2Weights, numHidden2Nodes, numOutputNodes);
            updateWeights(hiddenLayer1Weights, numHidden2Nodes, numHidden1Nodes, hiddenLayer1, hiddenLayer2Error, learningRate);

            getHiddenError(hiddenLayer1Error, hiddenLayer1, hiddenLayer2Error, hiddenLayer1Weights, numHidden1Nodes, numHidden2Nodes);
            updateWeights(inputLayerWeights, numHidden1Nodes, numInputNodes, inputLayer, hiddenLayer1Error, learningRate);

        }

      }

    return 0;

}


// create matrix to store all weights for implementing matrix multiplication instead of dot product
// void createRandomInputWeightMatrix(float weightMatrix[][numInputNodes]) {
void getFloatArrayFromIntArray(float float_array[], int array[], int array_length) {

        for(int i = 0; i < array_length; i++) {
                float_array[i] = (float)array[i];
        }
}

void createRandomWeights(float weightArray[], int array_rows, int array_cols) {

        for(int i = 0; i < array_rows; i++) {
                for(int j = 0; j < array_cols; j++) {

                        weightArray[(i * array_cols) + j] = rand_weight();
                }
        }
}

void initializeTarget(float target[], int numberOnPicture) {

        int isEven = (numberOnPicture % 2 == 0) ? 1 : 0;

        target[0] = (isEven)? 1 : 0;

        target[1] = 0;
        // printf("target[0] = %f\n", target[0]);
        int primes[] = {0, 1, 2, 3, 5, 7};

        for (int i = 0; i < 6; i++) {
            if (numberOnPicture == primes[i]) {
                target[1] = 1;

                break;
            }
        }
        // printf("target[1] = %f\n", target[1]);

}

// calculate output by matrix multiplication of inputs and weights
// and then squash output
void getOutput(float output[], float input[], float weightArray[], int array_rows, int array_cols) {

        float sumMultiplicationResult;

        for(int i = 0; i < array_rows; i++) {
                for(int j = 0; j < array_cols; j++) {

                        sumMultiplicationResult += input[j] * weightArray[(i * array_cols) + j];
                }

                output[i] = sumMultiplicationResult;
                sumMultiplicationResult = 0;
        }

}

void squashOutput(float output[], int array_length) {

        for(int i = 0; i < array_length; i++) {
                output[i] = 1.0 / (1.0 + pow(M_E, -1 * output[i]));
                // printf("squashed output[%d] = %f\n", i, output[i]);
        }
}


void getOutputError(float error[], float target[], float output[], int array_length) {

        for (int i = 0; i < array_length; i++) {
                error[i] = (target[i] - output[i]) * (output[i] * (1.0 - output[i]));

                // printf("outputError[%d] = %f\n", i, error[i]);
        }
}


void getHiddenError(float error[], float hiddenLayer[], float prev_error[], float hiddenWeights[], int firstLayerLength, int secondLayerLength) {

        for (int i = 0; i < firstLayerLength; i++) {

                float sumMultiplicationResult = 0;

                for (int j = 0; j < secondLayerLength; j++) {

                        sumMultiplicationResult += prev_error[j] * hiddenWeights[(j * firstLayerLength) + i];
                }

                error[i] = (hiddenLayer[i] * (1.0 - hiddenLayer[i])) * sumMultiplicationResult;
        }

}


float getAverageErrorFromTarget(float target[], float outputLayer[], int array_length) {

        float allErrorSum = 0;

        for (int i = 0; i < array_length; i++) {

            allErrorSum += powf((target[i] - outputLayer[i]), 2);

        }

        return (allErrorSum / array_length);
    }

int testOuput(float outputLayer[], int numberOnPicture, int array_length) {

        int index_of_max = 0;

        if (outputLayer[1] > outputLayer[0]) {
            index_of_max = 1;
        }

        int isEven = (numberOnPicture % 2 == 0) ? 1 : 0;

        return (index_of_max == isEven) ? 0 : 1;
}


void updateWeights(float weightArray[], int array_rows, int array_cols, float layer[], float error[], float learningRate) {

        float deltaWeightArray[array_rows * array_cols];

        for (int i = 0; i < array_rows; i++) {
                for (int j = 0; j < array_cols; j++) {

                        deltaWeightArray[i * array_cols + j] = error[i] * layer[j] * learningRate;

                        weightArray[i * array_cols + j] += deltaWeightArray[i * array_cols + j];
                }
        }

}
