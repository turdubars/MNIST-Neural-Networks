/**************************************************************************
*
*   A CRUDE LIBRARY FOR LOADING MNIST DATA - BY MICHAEL BRADY
*
**************************************************************************/

#ifndef __MNIST_H__
#define __MNIST_H__

#include <stdio.h>      // needed for printf statements
#include <string.h>     // needed for draw_input (strcopy)

// define the data structure
typedef struct mnist_data {
	double data[28][28];        // 28x28 of 1s or 0s for the image
	int label;                  // label : 0 to 9
} mnist_data;

/*************************************************************************/
// CONVERT BINARY TO INT (0 TO 255)
static unsigned int mnistBin2Int(char *v)
{
	int i;
	unsigned int ret = 0;

	for (i = 0; i < 4; ++i) {
		ret <<= 8;
		ret |= (unsigned char)v[i];
	}

    return ret;
}

/*************************************************************************/
// vType = 0, load full training set
// vType = 1, load full testing data
// vType = 2, load 10 pre-selected images from training set
int mnistLoad(mnist_data **pData, unsigned int *count, int vType)
{
	char tmp[4]; // size of an integer

	unsigned int image_cnt, label_cnt;
	unsigned int image_dim[2];

    FILE *ifp;
    FILE *lfp;

    if(!(vType == 0 || vType == 1 || vType == 2)){
        printf("incorrect data load type\n");
        return -1;
    }

    // if vType = 1, load testing data,
    // if vType = 0 or 2 load training data:
    if (vType == 1){
        ifp = fopen("mnist/t10k-images.idx3-ubyte", "rb");
        lfp = fopen("mnist/t10k-labels.idx1-ubyte", "rb");
    } else {
        ifp = fopen("mnist/train-images.idx3-ubyte", "rb");
        lfp = fopen("mnist/train-labels.idx1-ubyte", "rb");
    }

    // do some error checking
	if (!ifp || !lfp) {
        printf("error: MNIST files not found\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -1;
	}

    // read ifp header, error check
	fread(tmp, 1, 4, ifp);
	if (mnistBin2Int(tmp) != 2051) {
        printf("the image file doesn't have a valid header\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -2;
	}

    // read lfp header, error check
	fread(tmp, 1, 4, lfp);
	if (mnistBin2Int(tmp) != 2049) {
        printf("the label file doesn't have a valid header\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -3;
	}

    // reading more of the image header (number of images)
	fread(tmp, 1, 4, ifp);
	image_cnt = mnistBin2Int(tmp);

    // read more of the label header (number of labels)
	fread(tmp, 1, 4, lfp);
	label_cnt = mnistBin2Int(tmp);

    // check if files sizes match
	if (image_cnt != label_cnt) {
        printf("file sizes do not match\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -4;
	}

    // finish reading in image file header - dimensions of the image file
	for (int i = 0; i < 2; ++i) {
		fread(tmp, 1, 4, ifp);
		image_dim[i] = mnistBin2Int(tmp);
	}

    // check that the image files are 28 x 28
	if (image_dim[0] != 28 || image_dim[1] != 28) {
        printf("header says image files are not 28 x 28\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -2;
	}

    if (vType != 2){ // if normally loading files
        *count = image_cnt;
        *pData = (mnist_data *)malloc(sizeof(mnist_data) * image_cnt);
        for (unsigned int i = 0; i < image_cnt; ++i) {
            unsigned char read_data[28 * 28]; // read data
            mnist_data *d = &(*pData)[i];
            fread(read_data, 1, 28*28, ifp);

            for (int j = 0; j < 28*28; ++j) {
                d->data[j/28][j%28] = read_data[j] / 255.0;
            }

            fread(tmp, 1, 1, lfp);
            d->label = tmp[0];
        }
    } else { // if only saving ten sample files to the data struct
        int gSamples[10] = {6, 7, 56, 58, 90, 173, 213, 214, 226, 245};
        *count = 10;
        *pData = (mnist_data *)malloc(sizeof(mnist_data) * 10);
        int idx = 0;
        for (unsigned int i = 0; i < image_cnt; ++i) {
            unsigned char read_data[28 * 28];
            mnist_data *d = &(*pData)[idx];
            fread(read_data, 1, 28*28, ifp); // read in image data
            fread(tmp, 1, 1, lfp);           // read in label data

            if(gSamples[idx] == i){ // only save the data from gSamples[]
                for (int j = 0; j < 28*28; ++j) {
                    d->data[j/28][j%28] = read_data[j] / 255.0;
                }
                d->label = tmp[0];
                idx++;
            }
        }
    }

	if (ifp) fclose(ifp);
	if (lfp) fclose(lfp);

	return 0;
}

/*************************************************************************
// DRAW THE CHARACTER
int draw_label(mnist_data *pData, int idx)
{
    char pixel[1];
    double val;
    printf("label = %d:\n", pData[idx].label);
    for(int i = 0; i<28; i++){
        for(int j = 0; j<28; j++){
            val = pData[idx].data[i][j];
            if ( val > .1) strcpy(pixel, "X");
            else strcpy(pixel, " ");
            printf("%s", pixel);
        }
        printf("\n");
    }

    return 0;
}

/*************************************************************************/
// CONVERT IMAGE DATA (28 X 28 PIXELS) INTO A SINGLE VECTOR
// ALSO CONVERTS THE FLOATING POINT PIXELS INTO BINARY (ON/OFF) VALUES
// ALSO ADDS A BIAS NODE TO THE BEGINNING OF THE VECTOR
// MAY ALSO ADD NOISE TO THE OUTPUT VECTOR
int get_input(int zVector[], mnist_data *data, int zLabel, float noise)
{
    char pixel[1];
    double val;

    // set first element of vector to 1 (for bias node)
    zVector[0] = 1;

    // set rest of vector elements to the data of the image to be learned
    for(int i = 0; i<28; i++){
        for(int j = 0; j<28; j++){
            val = data[zLabel].data[i][j];
            // make data binary before saving to the vector
            if ( val > .1) zVector[(i*28)+j+1] = 1;
            else zVector[(i*28)+j+1] = 0;
        }
    }

    // randomly flip some of the pixels depending on noise parameter
    for(int i=1; i<785; i++){ // don't flip the first node, the bias node
        float randFlip = rand_frac();
        if (randFlip < noise) zVector[i] = abs(zVector[i]-1);
    }

    return 0;
}

/*************************************************************************/
// DRAW VECTOR TO SCREEN WITH ITS LABEL
int draw_input(int zVector[], int zLabel)
{
    char pixel[1];
    double val;
    printf("correct category: %d\n", zLabel);
    for(int i = 0; i<28; i++){
        for(int j = 0; j<28; j++){
            val = zVector[(i*28)+j+1];
            if ( val == 1) strcpy(pixel, "X");
            else strcpy(pixel, " ");
            printf("%s", pixel);
        }
        printf("\n");
    }
    printf("\n\n");
    return 0;
}

/*************************************************************************/
#endif /*__MNIST_H__*/
