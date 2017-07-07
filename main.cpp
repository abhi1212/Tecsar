
//   We can Use Thrust library 

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
size_t numRows() { return (size_t) imageInputRGBA.rows; }
size_t numCols() { return (size_t) imageInputRGBA.cols; }


/********************************************************Function Definitions********************************************************************************************/

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char * d_redBlurred,
                        unsigned char * d_greenBlurred,
                        unsigned char * d_blueBlurred,
                        const int filterWidth, const int tilesize, const int s, const int oRow, const int oCol);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
const float* const h_filter, const size_t filterWidth);

void seperate_channel(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                      const size_t numRows, const size_t numCols,
                      const int oRow,const int oCol);


void recombine_channels(unsigned char * const d_outputImageRGBA,
			unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
			const size_t numRows, const size_t numCols,
			const int oRow,const int oCol);

void pool(unsigned char *image, unsigned char *output, int oRow,int oCol,int fsize,int stride);


/********************************************************Main Function********************************************************************************************/
	//Changed the type of All output images to unsigned char..  



int main(int argc, char **argv)
{
	uchar4  *d_inputImageRGBA,*h_inputImageRGBA;	//uchar4 is a structure with 4 fields	      	
	unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred,*d_outputImageRGBA,*h_outputImageRGBA;
	float *d_filter,*h_filter;
	int i;
	/* Defined a structure with 96 output arrays for first layer*/
	
	struct outer{
	unsigned char *h_outputImageRGBA;
	} out[96];

	struct d_outer{
	unsigned char *d_outputImageRGBA;
	} d_out[96];

	struct d_pool{
	unsigned char *d_poolImageRGBA;
	} d_pool[96];

	
	/*************************************************************/ 

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

	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4); //allocate memory for the output //Here we are allocating the mempry equal to input image size.




  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
}


/*********************************************************Setting the Kernel Weights**********************************************************/

	h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);    //Creating an Array for Input Image

		
	/*h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);   //Creating an Array for Output Image
	h_outputImageRGBA1 = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);   //Creating an Array for Output Image*/
	


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

	//Let the output image also be unsigned char..........


/**********************************************************First Layer Pooling Layer Variables*******************************************************************/

	const int oCol=(numCols()-filterWidth)/s+1;	//Number of Output Columns after stride and filter
	const int oRow=(numRows()-filterWidth)/s+1;
	const int oNumPixels=oCol*oRow;			//Total Number of Pixels


	int fsize=3;
	int stride=3;

	int output_rows=((oRow-fsize)/stride) +1;
  	int output_columns=((oCol-fsize)/stride) +1;	
	int pool_pixels= (output_rows*output_columns);
	

	/*Malloced Input Image and Output Image, Memset done and also Memcpy*/


	for(i=0;i<96;i++)
	{
		out[i].h_outputImageRGBA=(unsigned char *)malloc(sizeof(unsigned char) *oNumPixels);	
		memset(out[i].h_outputImageRGBA,0,sizeof(unsigned char) *oNumPixels);
		
	}

	for(i=0;i<96;i++)
	{
		d_pool[i].d_poolImageRGBA=(unsigned char *)malloc(sizeof(unsigned char) *pool_pixels);
		memset(d_pool[i].d_poolImageRGBA,0,sizeof(unsigned char) *pool_pixels);	
	}

  
	checkCudaErrors(cudaMalloc((void**)&d_inputImageRGBA, sizeof(uchar4) * numPixels)); 	//Numpixels size of original image.

	
	for(i=0;i<96;i++)
	{
		checkCudaErrors(cudaMalloc((void**)&d_out[i].d_outputImageRGBA,(sizeof(unsigned char) * oNumPixels)));	//ONumpixels size after stride
	}

	//checkCudaErrors(cudaMalloc((void**)&d_outputImageRGBA, sizeof(uchar4) * oNumPixels));	//ONumpixels size after stride


	for(i=0;i<96;i++)
	{
		checkCudaErrors(cudaMemset(d_out[i].d_outputImageRGBA, 0, sizeof(unsigned char) * oNumPixels)); //make sure no memory is left laying around
	}



	checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(int) * numPixels, cudaMemcpyHostToDevice)); //Input Memcpy
	checkCudaErrors(cudaMalloc((void**)&d_redBlurred,    sizeof(unsigned char) * oNumPixels));			  //Malloc 3 channels
	checkCudaErrors(cudaMalloc((void**)&d_greenBlurred,  sizeof(unsigned char) * oNumPixels));
	checkCudaErrors(cudaMalloc((void**)&d_blueBlurred,   sizeof(unsigned char) * oNumPixels));
	checkCudaErrors(cudaMemset(d_redBlurred,   0, sizeof(unsigned char) * oNumPixels));
	checkCudaErrors(cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * oNumPixels));
	checkCudaErrors(cudaMemset(d_blueBlurred, 0, sizeof(unsigned char) * oNumPixels));


	allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);	//Mallocing output of 3 channels and also filters.

	printf("Size of uchar4 is %d\n",sizeof(uchar4)); // Its 4 byte Long
 
	printf("Size of size_t is %d\n",sizeof(uchar4)); // Its 4 byte Long

	GpuTimer timer;		//Start the Timer
	timer.Start(); 

	

	/**********************************First Layer Alex Net**********************************************************************/
  
	seperate_channel(h_inputImageRGBA, d_inputImageRGBA, numRows(), numCols(),oRow,oCol);    // Call Seperate channel

	for(i=0;i<96;i++)
	{


	
	your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA,numRows(), numCols(), d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth, tilesize, s, oRow, oCol);     // 																Convolution First Layer 96 kernels of size 11*11*3
	
	recombine_channels(d_out[i].d_outputImageRGBA, d_redBlurred, d_greenBlurred, d_blueBlurred,numRows(), numCols(),oRow,oCol);


	//cudaDeviceSynchronize();	

	checkCudaErrors(cudaMemcpy((out[i].h_outputImageRGBA),d_out[i].d_outputImageRGBA, sizeof(unsigned char) * oNumPixels, cudaMemcpyDeviceToHost));

	}

	//pool(d_out[20].d_outputImageRGBA, d_pool[20].d_poolImageRGBA, oRow, oCol, fsize, stride);



	//pool_firstlayer();      Pooling Layer (size 3*3) (stride 2*2)

	//relu();
	

	/******************************************************************************************************************************/


	timer.Stop();
	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());
	
	


	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());


	if (err < 0) {
	//Couldn't print! Probably the student closed stdout - bad news
	std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
	exit(1);
	}


	checkCudaErrors(cudaMemcpy(out[90].h_outputImageRGBA, d_out[90].d_outputImageRGBA, sizeof(unsigned char) *oNumPixels, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(out[90].h_outputImageRGBA, d_outputImageRGBA, sizeof(unsigned char) *oNumPixels, cudaMemcpyDeviceToHost));

	/*cv::Mat output(oRow, oCol, CV_8UC4, h_outputImageRGBA);

	cv::Mat imageOutputBGR;
	cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);


	  //output the image
	cv::imwrite(output_file.c_str(), imageOutputBGR);

	checkCudaErrors(cudaFree(d_inputImageRGBA));
	checkCudaErrors(cudaFree(d_outputImageRGBA));
	checkCudaErrors(cudaFree(d_filter));*/
	

  return 0;
}

