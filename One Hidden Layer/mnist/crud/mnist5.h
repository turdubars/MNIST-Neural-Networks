/**************************************************************************
*
*   A C library for loading MNIST data for Assignment 1 by Michael Brady
*
**************************************************************************/

#ifndef __MNIST_H__
#define __MNIST_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// define the data structure
typedef struct mnist_data {
	double data[28][28];            // 28x28 of 1s or 0s for the image
	unsigned int label;             // label : 0 to 9
} mnist_data;


// convert binary to int (0 to 255)
static unsigned int mnist_bin_to_int(char *v)
{
	int i;
	unsigned int ret = 0;

	for (i = 0; i < 4; ++i) {
		ret <<= 8;
		ret |= (unsigned char)v[i];
	}
	return ret;
}

// load the file and create the data structure
int mnist_load(mnist_data **data, unsigned int *count)
{
	char tmp[4];

	unsigned int image_cnt, label_cnt;
	unsigned int image_dim[2];

	FILE *ifp = fopen("mnist/train-images.idx3-ubyte", "rb");
	FILE *lfp = fopen("mnist/train-labels.idx1-ubyte", "rb");

    // do some error checking
	if (!ifp || !lfp) {
        printf("error: one or both of the MNIST files not found\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -1;
	}

    // read ifp header, error check
	fread(tmp, 1, 4, ifp);
	if (mnist_bin_to_int(tmp) != 2051) {
        printf("the image file doesn't have a valid header\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -2;
	}

    // read lfp header, error check
	fread(tmp, 1, 4, lfp);
	if (mnist_bin_to_int(tmp) != 2049) {
        printf("the label file doesn't have a valid header\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -3;
	}

    // reading more of the image header (number of images)
	fread(tmp, 1, 4, ifp);
	image_cnt = mnist_bin_to_int(tmp);

    // read more of the label header (number of labels)
	fread(tmp, 1, 4, lfp);
	label_cnt = mnist_bin_to_int(tmp);

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
		image_dim[i] = mnist_bin_to_int(tmp);
	}

    // check that the image files are 28 x 28
	if (image_dim[0] != 28 || image_dim[1] != 28) {
        printf("header says image files are not 28 x 28\n");
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
		return -2;
	}

	*count = image_cnt;
	*data = (mnist_data *)malloc(sizeof(mnist_data) * image_cnt);
	for (int i = 0; i < image_cnt; ++i) {
		unsigned char read_data[28 * 28];
		mnist_data *d = &(*data)[i];
		fread(read_data, 1, 28*28, ifp);

		for (int j = 0; j < 28*28; ++j) {
            if (read_data[j] > 25)
//                d->data[j/28][j%28] = 1.0;
//            else
//                d->data[j/28][j%28] = 0.0;
			d->data[j/28][j%28] = read_data[j] / 255.0;
		}

		fread(tmp, 1, 1, lfp);
		d->label = tmp[0];
	}
    
	if (ifp) fclose(ifp);
	if (lfp) fclose(lfp);

	return 0;
}

// get the (noisy) training set in random order
int draw_label(mnist_data *data, int theLabel)
{
    char pixel[1];
    double val;
    printf("label = %d:\n", data[theLabel].label);  
    for(int i = 0; i<28; i++){
        for(int j = 0; j<28; j++){
            val = data[theLabel].data[i][j];
            if ( val > .1) strcpy(pixel, "X");
            else strcpy(pixel, " ");
            printf("%s", pixel);
        }
        printf("\n");
    }
}





#endif /* __MNIST_H__ */