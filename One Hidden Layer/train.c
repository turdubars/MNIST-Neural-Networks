#include <stdio.h>
#include <math.h>
#include "randlib.h"
#include "mnist/mnist.h"

#define numInputNodes 785
#define numHiddenNodes 21
#define numOutputNodes 785
#define numEpochs 5

void getFloatArrayFromIntArray(float float_array[], int array[], int array_length);
void createRandomWeights(float weightArray[], int rows, int cols);
void initializeTarget(float target[], int input[], int numberOnPicture);
void getOutput(float output[], float input[], float weightArray[], int array_rows, int array_cols);
void squashOutput(float output[], int array_length);
void getOutputError(float error[], float target[], float output[], int array_length);
void getHiddenError(float error[], float hiddenLayer[], float prev_error[], float hiddenWeights[], int firstLayerLength, int secondLayerLength);
float getAverageErrorFromTarget(float target[], float outputLayer[], int array_length);
float getAverageErrorFromOutputErrors(float outputLayerError[], int array_length);
void updateWeights(float weightArray[], int array_rows, int array_cols, float layer[], float error[], float learningRate);
int getIsEvenError(float outputLayer[], float target[]);
float getAverageRoundedErrorFromTarget(float target[], float outputLayer[], int array_length);



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

    mnist_data *zData;      // each image is 28x28 pixels
    unsigned int sizeData;  // depends on loadType
    int loadType = 1; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }

    mnist_data *zData1;      // each image is 28x28 pixels
    unsigned int sizeData1;  // depends on loadType
    int loadType1 = 2; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData1, &sizeData1, loadType1)){
        printf("something went wrong loading data set\n");
        return -1;
    }

    float learningRate = 0.05;

    // creating variable to store input values
    int inputVec[numInputNodes];

    float hiddenLayer[numHiddenNodes];

    // create array to store outputs
    float outputLayer[numOutputNodes];

    float target[numOutputNodes];

    float inputLayer[numInputNodes];

    float outputLayerError[numOutputNodes];

    float hiddenLayerError[numHiddenNodes];

    // 0 = 1
    // 1 = 3
    // 2 = 0
    // 3 = 4
    // 4 = 6
    // ...
    float average_error;

    float outputResults[numEpochs][4];

    float inputLayerWeights[(numHiddenNodes) * numInputNodes];
    createRandomWeights(inputLayerWeights, numHiddenNodes, numInputNodes);

    float hiddenLayerWeights[numOutputNodes * numHiddenNodes];
    createRandomWeights(hiddenLayerWeights, numOutputNodes, numHiddenNodes);

    // loadType = 0, 60k training images
    // loadType = 1, 10k testing images
    // loadType = 2, 10 specific images from training set

    for(int epoch = 0; epoch < numEpochs; epoch++) {

        float average_error_from_target = 0;
        float average_rounded_error_from_target = 0;
        float average_error_from_error_array = 0;
        float is_even_error = 0;

        for(int testImage = 0; testImage < sizeData1    ; testImage++) {

          get_input(inputVec, zData1, testImage, sampNoise);
          // draw_input(inputVec, zData[testImage].label);

          initializeTarget(target, inputVec, zData1[testImage].label);

          // function for getting output needs float arrays,
          // so we need to convert the type of our inputVec to float
          getFloatArrayFromIntArray(inputLayer, inputVec, numInputNodes);

          // calculate nodes for a hidden layer and load it to hiddenLayer array
          getOutput(hiddenLayer, inputLayer, inputLayerWeights, numHiddenNodes, numInputNodes);

          squashOutput(hiddenLayer, numHiddenNodes);
          hiddenLayer[numHiddenNodes - 1] = 1; // let the last node of the hidden layer to be a bias and set it to 1

          // calculate nodes for output layer and load it to outputLayer array
          getOutput(outputLayer, hiddenLayer, hiddenLayerWeights, numOutputNodes, numHiddenNodes);

          squashOutput(outputLayer, numOutputNodes);

          average_error_from_target += getAverageErrorFromTarget(target, outputLayer, numOutputNodes);

          // the same as "average_error_from_target" just rounds the values in the outputLayer
          average_rounded_error_from_target += getAverageRoundedErrorFromTarget(target, outputLayer, numOutputNodes);

          getOutputError(outputLayerError, target, outputLayer, numOutputNodes);
          average_error_from_error_array += getAverageErrorFromOutputErrors(outputLayerError, numOutputNodes);

          is_even_error += getIsEvenError(outputLayer, target);
        }

        // storing outputs in the outputResults for printing outputs in finished form
        outputResults[epoch][0] = (average_error_from_target / sizeData1);
        outputResults[epoch][1] = (average_rounded_error_from_target / sizeData1);
        outputResults[epoch][2] = (average_error_from_error_array / sizeData1);
        outputResults[epoch][3] = (is_even_error / sizeData1);

        for(int pictureIndex = 0; pictureIndex < sizeData; pictureIndex++) {
;
            get_input(inputVec, zData, pictureIndex, sampNoise);
            // draw_input(inputVec, zData[pictureIndex].label);

            initializeTarget(target, inputVec, zData[pictureIndex].label);

            getFloatArrayFromIntArray(inputLayer, inputVec, numInputNodes);

            getOutput(hiddenLayer, inputLayer, inputLayerWeights, numHiddenNodes, numInputNodes);

            squashOutput(hiddenLayer, numHiddenNodes);
            hiddenLayer[numHiddenNodes - 1] = 1;

            // for(int i = 0; i < numHiddenNodes; i++) {
            //
            //     printf("hidden[%d] = %f\n", i, hiddenLayer[i]);
            // }

            getOutput(outputLayer, hiddenLayer, hiddenLayerWeights, numOutputNodes, numHiddenNodes);

            squashOutput(outputLayer, numOutputNodes);

            // for(int k = 0; k < 785; k++) {
            //
            //     printf("%d", outputLayer[k] > 0.5 ? 1:0);
            //     if (k==0 || k%28==0) {
            //         printf("\n");
            //     }
            // }

            getOutputError(outputLayerError, target, outputLayer, numOutputNodes);

            getHiddenError(hiddenLayerError, hiddenLayer, outputLayerError, hiddenLayer, numHiddenNodes, numOutputNodes);

            updateWeights(hiddenLayerWeights, numOutputNodes, numHiddenNodes, hiddenLayer, outputLayerError, learningRate);

            updateWeights(inputLayerWeights, numHiddenNodes, numInputNodes, inputLayer, hiddenLayerError, learningRate);

        }


      }


        // printf("error_per_epoch[%d] = %f\n", epoch, (error_per_epoch / sizeData));

        printf("average_error_from_target,average_rounded_error_from_target,average_error_from_error_array,bias_node_error\n");

        for (int i = 0; i < numEpochs; i++) {
            for (int j = 0; j < 4; j++) {

                printf("%f", outputResults[i][j]);

                if (j == 3) {
                    printf("\n");
                } else {
                    printf(", ");
                }
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

void initializeTarget(float target[], int input[], int numberOnPicture) {

        for(int i = 0; i < numOutputNodes; i++) {
                target[i] = (float)input[i];
        }

        // set target for the first node if the number on the picture is odd = 0 or even = 1
        // target[0] = (numberOnPicture % 2 == 0) ? 1 : 0;

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

                        sumMultiplicationResult += prev_error[j] * hiddenWeights[j * firstLayerLength + i];
                }

                error[i] = (hiddenLayer[i] * (1.0 - hiddenLayer[i])) * sumMultiplicationResult;
        }

}


float getAverageRoundedErrorFromTarget(float target[], float outputLayer[], int array_length) {

        float allErrorSum = 0;

        for (int i = 0; i < array_length; i++) {

                float roundedNum = round(outputLayer[i]);

                allErrorSum += (roundedNum == target[i])? 0 : 1;
        }

        return (allErrorSum / array_length);
}

float getAverageErrorFromTarget(float target[], float outputLayer[], int array_length) {

    float allErrorSum = 0;

    for (int i = 0; i < array_length;    i++) {

        allErrorSum += powf((target[i] - outputLayer[i]), 2);

    }

    return (allErrorSum / array_length);
}

float getAverageErrorFromOutputErrors(float outputLayerError[], int array_length) {

        float allErrorSum = 0;

        for (int i = 0; i < array_length; i++) {


                allErrorSum += fabs(outputLayerError[i]);
        }

        return (allErrorSum / array_length);
}

int getIsEvenError(float outputLayer[], float target[]) {

    float roundedNum = round(outputLayer[0]);
    // printf("isEvenOutput === %d\n\n", roundedNum);

    if (roundedNum == target[0]) {

        return 0;
    }
    else {

        return 1;
    }
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
