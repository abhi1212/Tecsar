
#include "utils.h"
#include <stdio.h>


__global__ void gaussian_blur(const unsigned char* const inputChannel,
                    unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth,const int s, int oRows, int oCols)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
 
    
int x=blockIdx.x * blockDim.x + threadIdx.x;
    int y=blockIdx.y * blockDim.y + threadIdx.y;
    const int thread_1D_pos = y * oCols + x;

   if ( x>=oCols ||y>=oRows )
   {
       return;
   }
   float sum=0.0f;
//printf("gloc=%d =>threadId.x=%d,threadId.y=%d,blockId.x=%d,blocId.y=%d,position.x=%d,position.y=%d\n",thread_1D_pos,threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,threadIdx.x +(blockIdx.x* blockDim.x),threadIdx.y +(blockIdx.y* blockDim.y));
int kidx=0;
   for(int r=0; r<filterWidth;++r){
        for(int c=0; c<filterWidth;++c){
        
            int idx=(y*s+r)*numCols+x*s+c;
            
        float filter_value=filter[kidx++];
        sum+=filter_value*static_cast<float>(inputChannel[idx]);
   
        }
    }
    outputChannel[thread_1D_pos]=sum;
    
  
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}


//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO
  //
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  // Basically make int2 creates a structure of 2 fields and will initiliaze both of them..

  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);  //  it constructs a vector with value x, y., 
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;	 //  
  const int absolute_image_position_x = thread_2D_pos.x;
  const int absolute_image_position_y = thread_2D_pos.y;
  if ( absolute_image_position_x >= numCols ||
       absolute_image_position_y >= numRows )
  {
      return;
  }
  redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
  greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
  blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
//if(thread_1D_pos==500)
	//printf("in kernel value.x=%f\t,value.y=%f\tvalue.z=%f",static_cast<float>(outputImageRGBA[500].x),static_cast<float>(outputImageRGBA[500].y),static_cast<float>(outputImageRGBA[500].z));
}



unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter, sizeof( float) * filterWidth * filterWidth));
  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

}


void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth,const int tilesize,const int s,const int oRow,const int oCol)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(32,2);


  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(oCol/blockSize.x+1,oRow/blockSize.y+1);

  /*TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                              numRows,
                                              numCols,
                                              d_red,
                                              d_green,
                                              d_blue);
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());*/

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_red,
                                         d_redBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth,s,oRow,oCol);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<gridSize, blockSize>>>(d_green,
                                         d_greenBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth,s,oRow,oCol);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<gridSize, blockSize>>>(d_blue,
                                         d_blueBlurred,
                                         numRows,
                                         numCols,
                                         d_filter,
                                         filterWidth,s,oRow,oCol);

  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}




void seperate_channel(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                      const size_t numRows, const size_t numCols,
                      const int s,const int oRow,const int oCol)
{
	 
	const dim3 blockSize(32,2);
	const dim3 gridSize(oCol/blockSize.x+1,oRow/blockSize.y+1);


  //TODO: Launch a kernel for separating the RGBA image into different color channels
	separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                              numRows,
                                              numCols,
                                              d_red,
                                              d_green,
                                              d_blue);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	
}



void conv_firstlayer()
{





}


















