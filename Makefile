NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
CUDA_INCLUDEPATH=/usr/local/cuda-8.0/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS=-O3 -arch=compute_50 -Xcompiler -Wall -Xcompiler -Wextra 

GCC_OPTS=-O3 -Wall -Wextra 

student: main.o ConvTraditional.o Makefile
	$(NVCC) -o conv main.o ConvTraditional.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS) `pkg-config opencv --cflags --libs`

main.o: main.cpp timer.h utils.h 
	g++ -c main.cpp $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -I $(CUDA_INCLUDEPATH) 

ConvTraditional.o: ConvTraditional.cu  utils.h
	nvcc -c ConvTraditional.cu $(NVCC_OPTS) 


clean:
	rm -f *.o *.png hw
