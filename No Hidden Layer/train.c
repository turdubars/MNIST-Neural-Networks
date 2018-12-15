#include <stdio.h>
#include <math.h>
#include "randlib.h"
#include "mnist/mnist.h"

#define numInputNodes 785
#define numOutputNodes 10

void initializeTarget(float target[], int numberOnPicture);
void createRandomWeightMatrix(float weightMatrix[][numInputNodes]);
void getOutput(float output[], int inputVec[], float weightMatrix[][numInputNodes]);
void squashOutput(float output[]) ;
void getError(float error[], float target[], float output[]);
float getAverageError(float error[]);
void updateWights(float weightMatrix[][numInputNodes], int inputVec[], float error[], float learningRate);

int main(int argc, char const *argv[]) {
  // --- an example for working with random numbers
  seed_randoms();
  // float sampNoise = rand_frac()/2.0; // sets default sampNoise
  float sampNoise = 0;

  // --- a simple example of how to set params from the command line
  if(argc == 2){ // if an argument is provided, it is SampleNoise
      sampNoise = atof(argv[1]);
      if (sampNoise < 0 || sampNoise > .5){
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

  // creating variable to store input values
  int inputVec[numInputNodes];


  float target[numOutputNodes];

  // create array to store outputs
  float output[numOutputNodes];

  float error[numOutputNodes];
  // 0 = 1
  // 1 = 3
  // 2 = 0
  // 3 = 4
  // 4 = 6
  // ...

  float weightMatrix[numOutputNodes][numInputNodes];
  createRandomWeightMatrix(weightMatrix);


  // loadType = 0, 60k training images
  // loadType = 1, 10k testing images
  // loadType = 2, 10 specific images from training set

for(float learningRate = 0.1; learningRate >= 0.00001; learningRate /= 10) {


  createRandomWeightMatrix(weightMatrix);

  for(int epochs = 0; epochs < 10; epochs++) {

    for(int pictureIndex = 0; pictureIndex < sizeData; pictureIndex++){
        get_input(inputVec, zData, pictureIndex, sampNoise);
        // draw_input(inputVec, zData[pictureIndex].label);

        initializeTarget(target, zData[pictureIndex].label);

        getOutput(output, inputVec, weightMatrix);

        squashOutput(output);

        getError(error, target, output);

        updateWights(weightMatrix, inputVec, error, learningRate);
    }

    float theError = 0;

    for(int testImage = 0; testImage < sizeData1; testImage++) {

      get_input(inputVec, zData1, testImage, sampNoise);
      draw_input(inputVec, zData[testImage].label);
      initializeTarget(target, zData1[testImage].label);

      getOutput(output, inputVec, weightMatrix);

      squashOutput(output);

      for (int i = 0; i < numOutputNodes; i++) {
          printf("output[%d] = %f\n", i, output[i]);
      }

      getError(error, target, output);

      theError += getAverageError(error);

    }
    printf("%f, ", (theError / sizeData1));
  }
  printf("\n");


}

  return 0;

}


// create matrix to store all weights for implementing matrix multiplication instead of dot product
void createRandomWeightMatrix(float weightMatrix[][numInputNodes]) {

    for (int i = 0; i < numOutputNodes; i++)
    {
      for (int j = 0; j < numInputNodes; j++)
      {
        // set random values for weights between -1 to 1
        weightMatrix[i][j] = rand_weight();
      }
    }
  }


void initializeTarget(float target[], int numberOnPicture) {

  for(int i = 0; i < numOutputNodes; i++) {
    if (i == numberOnPicture) {
      target[i] = 1;
    } else {
      target[i] = 0;
    }
    // printf("target[%d] = %f\n", i, target[i]);
  }
}

// calculate output by matrix multiplication of inputs and weights
// and then squash output
void getOutput(float output[], int inputVec[], float weightMatrix[][numInputNodes]) {

  // variable to store results of matrix multiplication
  float sumMultiplicationResult;

  for(int i = 0; i < numOutputNodes; i++) {
    sumMultiplicationResult = 0;
    for(int j = 0; j < numInputNodes; j++) {

      float multiplicationResult = inputVec[j] * weightMatrix[i][j];
      sumMultiplicationResult += multiplicationResult;
    }

    output[i] = sumMultiplicationResult;
    // printf("output[%d] = %f\n", i, sumMultiplicationResult);
  }
}

void squashOutput(float output[]) {

  for(int i = 0; i < numOutputNodes; i++) {
    output[i] = 1.0 / (1.0 + pow(M_E, -1 * output[i]));
    // printf("squashed output[%d] = %f\n", i, output[i]);
  }
}


// get error
void getError(float error[], float target[], float output[]) {

  for (int i = 0; i < numOutputNodes; i++) {
    error[i] = target[i] - output[i];

    // printf("error[%d] = %f\n", i, error[i]);
  }
}

float getAverageError(float error[]) {

  float allErrorSum = 0;
  for (int i = 0; i < numOutputNodes; i++) {
    allErrorSum += fabs(error[i]);
  }

  return (allErrorSum / numOutputNodes);
}

// calculate deltaWeight
// and update weight
void updateWights(float weightMatrix[][numInputNodes], int inputVec[], float error[], float learningRate) {

  float deltaWeightMatrix[numOutputNodes][numInputNodes];
  for(int i = 0; i < numOutputNodes; i++) {
    for (int j = 0; j < numInputNodes; j++) {
      deltaWeightMatrix[i][j] = inputVec[j] * error[i] * learningRate;
      // float temp = weightMatrix[i][j];
      weightMatrix[i][j] += deltaWeightMatrix[i][j];

    }
  }
}
