
/********************************************************Including Header Files********************************************************************************************/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdlib.h>
#include "timer.h"
#include "utils.h"
using namespace std;
using namespace cv;
Mat imageInputRGBA;
Mat imageOutputRGBA;
size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

/********************************************************Function Definitions********************************************************************************************/

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA,
                        const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth,const int tilesize,const int s,const int oRow,const int oCol);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
const float* const h_filter, const size_t filterWidth);

/********************************************************Main Function********************************************************************************************/

int main(int argc, char **argv)
{
	uchar4  *d_inputImageRGBA,*h_inputImageRGBA;
	uchar4 *h_outputImageRGBA, *d_outputImageRGBA;     //uchar4 is a structure with 4 fields	
	unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
	float *d_filter,*h_filter ;
 
	int filterWidth;
	int s;
	int tilesize;

	if(argc<4)  
	  { 
	     printf("Not enough Arguments\n 1.Filterwidth\n 2.Stride\n 3.Tilesize\n"); 	
	     return 0;
	  }

	filterWidth = atoi(argv[1]);
	printf("Filterwidth is %d\n",filterWidth);

	s = atoi(argv[2]);
	printf("Stride is %d\n",s);

	tilesize = atoi(argv[3]);
	printf("Tilesize is %d\n",tilesize);

	String output_file = "Conv_output.jpg";			//Output File Name
	String imageName( "index3.jpg" ); 			//Input File Name
     

    	Mat image;						//Declare the Mat Image 
        image = imread( imageName, CV_LOAD_IMAGE_COLOR ); 	// Read the file


    
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);         //Converts an image from one color space to another. http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html , Image is the source,  

	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4); //allocate memory for the output




  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
}


/*********************************************************Setting the Kernel Weights**********************************************************/

	h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);    //Creating an Array for Input Image
	h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);   //Creating an Array for Output Image
	h_filter = new float[filterWidth * filterWidth];                       //Creating an Array for Filter
	const size_t numPixels = numRows() * numCols();


	const float blurKernelSigma = 2.;
	float filterSum = 0.f; //for normalization

	for (int r = -filterWidth/2; r <= filterWidth/2; ++r) {
        	for (int c = -filterWidth/2; c <= filterWidth/2; ++c) {
      			float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			h_filter[(r + filterWidth/2) * filterWidth + c + filterWidth/2] = filterValue;
			filterSum += filterValue;
    		}
  	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -filterWidth/2; r <= filterWidth/2; ++r) {
	for (int c = -filterWidth/2; c <= filterWidth/2; ++c) {
	h_filter[(r + filterWidth/2) * filterWidth + c + filterWidth/2] *= normalizationFactor;
	    }
	}


/*************************************************************************************************************************************************/





	const int oCol=(numCols()-filterWidth)/s+1;	//Number of Output Columns after stride and filter
	const int oRow=(numRows()-filterWidth)/s+1;
	const int oNumPixels=oCol*oRow;			//Total Number of Pixels


	/*Malloced Input Image and Output Image, Memset done and also Memcpy*/
  

	checkCudaErrors(cudaMalloc((void**)&d_inputImageRGBA, sizeof(uchar4) * numPixels)); 	//Numpixels size of original image.
	checkCudaErrors(cudaMalloc((void**)&d_outputImageRGBA, sizeof(uchar4) * oNumPixels));	//ONumpixels size after stride
	checkCudaErrors(cudaMemset(d_outputImageRGBA, 0, sizeof(uchar4) * oNumPixels)); //make sure no memory is left laying around
	checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(int) * numPixels, cudaMemcpyHostToDevice)); //Input Memcpy
	checkCudaErrors(cudaMalloc((void**)&d_redBlurred,    sizeof(unsigned char) * oNumPixels));			  //Malloc 3 channels
	checkCudaErrors(cudaMalloc((void**)&d_greenBlurred,  sizeof(unsigned char) * oNumPixels));
	checkCudaErrors(cudaMalloc((void**)&d_blueBlurred,   sizeof(unsigned char) * oNumPixels));
	checkCudaErrors(cudaMemset(d_redBlurred,   0, sizeof(unsigned char) * oNumPixels));
	checkCudaErrors(cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * oNumPixels));
	checkCudaErrors(cudaMemset(d_blueBlurred, 0, sizeof(unsigned char) * oNumPixels));


	allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);	//Mallocing output of 3 channels and also filters.

	printf("Size of uchar4 is %d\n",sizeof(uchar4));
 
 
	/*Memory Allocation Done try to launch some Kernels*/
  
	GpuTimer timer;		//Start the Timer
	timer.Start();


	your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
				d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth,tilesize,s,oRow,oCol);

	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

	if (err < 0) {
	//Couldn't print! Probably the student closed stdout - bad news
	std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
	exit(1);
	}

	checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * oNumPixels, cudaMemcpyDeviceToHost));
	cv::Mat output(oRow, oCol, CV_8UC4, h_outputImageRGBA);

	cv::Mat imageOutputBGR;
	cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);


	  //output the image
	cv::imwrite(output_file.c_str(), imageOutputBGR);

	checkCudaErrors(cudaFree(d_inputImageRGBA));
	checkCudaErrors(cudaFree(d_outputImageRGBA));
	checkCudaErrors(cudaFree(d_filter));


  return 0;
}
