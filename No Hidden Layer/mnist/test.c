#include <stdio.h>
#include <math.h>
#include "randlib.h"
#include "mnist/mnist.h"


#define numInputNodes 785
#define numOutputNodes 10

int main(int argc, char const *argv[]) {
  // --- an example for working with random numbers
  seed_randoms();
  // float sampNoise = rand_frac()/2.0; // sets default sampNoise
  float sampNoise = 0.0;

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
  int loadType = 2; // loadType may be: 0, 1, or 2
  if (mnistLoad(&zData, &sizeData, loadType)){
      printf("something went wrong loading data set\n");
      return -1;
  }

  int inputVec[numInputNodes]; // creating variable to store input values
  int pictureIndex = 0;

  // 0 = 1
  // 1 = 3
  // 2 = 0
  // 3 = 4
  // 4 = 6
  // ...


  // variable to store results of matrix multiplication
  float multiplicationResult = 0;

  // load MNIST picture and convert it to a vector
  get_input(inputVec, zData, pictureIndex, sampNoise);

  // draw the picture with specified label
  draw_input(inputVec, zData[pictureIndex].label);



  return 0;
}
