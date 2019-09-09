#include "common.h"
#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#define BLOCK_SIZE 256   //max number of thread
#define BYTES_PER_PIXEL 3
#define STREAMS_CNT 4

using namespace cv;
using namespace std;


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number){
	if(err!=cudaSuccess){
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void multi_streams_kernel(unsigned char* input, unsigned char* output, int stream_idx, unsigned int pixels_per_stream ,unsigned int pixel_cnt) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned idx = stream_idx * pixels_per_stream + tid;

	if( tid < pixel_cnt ){
		unsigned int color_idx = idx * BYTES_PER_PIXEL;
		unsigned int gray_idx = idx;

		unsigned char blue	= input[color_idx];
		unsigned char green	= input[color_idx + 1];
		unsigned char red	= input[color_idx + 2];
		float gray = red * 0.3f + green * 0.59f + blue * 0.11f;
		output[gray_idx] = static_cast<unsigned char>(gray);
	}
}

int main()
{
    cv::Mat src_img = cv::imread("../data/test.jpg");//imread()函数载入图像
    if(src_img.empty())
    {
        fprintf(stderr, "Can not load image\n");//如果读入图像失败，返回错误信息
        return -1;
    }
    const int imgH = src_img.rows;//高度对应Y坐标
    const int imgW = src_img.cols;//宽度对应X坐标
    //输出图像，暂时初始化为和输入图像同尺寸的零值图像
    cv::Mat dst_img(imgH, imgW, CV_8UC1, Scalar(0));

    unsigned long int pixel_cnt = imgH * imgW;
    unsigned char *color_pixels, *gray_pixels;
    unsigned char *d_pixels_in, *d_pixels_out;
    color_pixels = (unsigned char *)malloc(sizeof(unsigned char) * pixel_cnt * BYTES_PER_PIXEL);
    gray_pixels = (unsigned char *)malloc(sizeof(unsigned char) * pixel_cnt);
    color_pixels = src_img.data;
	SAFE_CALL(cudaMalloc(&d_pixels_in, pixel_cnt * BYTES_PER_PIXEL * sizeof( unsigned char )), "Malloc colored device memory failed!");
	SAFE_CALL(cudaMalloc(&d_pixels_out, pixel_cnt * sizeof(unsigned char)), "Malloc grayed device memory failed!");

	// get the number of pixels each stream
	// unsigned pixels_per_stream = (pixel_cnt % STREAMS_CNT == 0 )? ( pixel_cnt/STREAMS_CNT ) : ( pixel_cnt/STREAMS_CNT + 1 );
	unsigned pixels_per_stream = (pixel_cnt + STREAMS_CNT - 1)/STREAMS_CNT;
	dim3 block(BLOCK_SIZE);
	dim3 grid((pixels_per_stream + BLOCK_SIZE - 1 )/BLOCK_SIZE);
	std::cout<<"GPU processing with multi streams......"<<std::endl;
    float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// crate streams for current device
    cudaStream_t* streams = (cudaStream_t*) malloc( STREAMS_CNT * sizeof( cudaStream_t ) );
    //cudaStream_t streams[STREAMS_CNT];

	for(int i=0; i<STREAMS_CNT; i++)
		SAFE_CALL(cudaStreamCreate(&streams[i]), "Create stream failed!");

	// pixel count in current stream
	// the number of pixels in last stream normally should be different with previous streams
	unsigned int pixel_in_cur_stream = 0;

	// start the stream execution for current device
	for( int i=0; i<STREAMS_CNT; i++ ){
		// this is the boundary check for the pixel number in last stream
		// normally, it coule not be the same number of pixels in each stream
		if ( i == STREAMS_CNT -1  )
			pixel_in_cur_stream = pixel_cnt - pixels_per_stream * (STREAMS_CNT - 1);
		else
			pixel_in_cur_stream = pixels_per_stream;
		// copy data from host to device
		SAFE_CALL(cudaMemcpyAsync(&d_pixels_in[i * pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof(unsigned char)],
								&color_pixels[i * pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof(unsigned char)],
								pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof(unsigned char),
								cudaMemcpyHostToDevice,
								streams[i]),
								"Device memory asynchronized copy failed!");
        

		// kernel launch
		multi_streams_kernel<<< grid, block, 0, streams[i] >>>(d_pixels_in, d_pixels_out, i, pixel_in_cur_stream, pixel_cnt);
		// copy data back from device to host
		SAFE_CALL(cudaMemcpyAsync(&gray_pixels[i * pixel_in_cur_stream ],
								&d_pixels_out[i * pixel_in_cur_stream ],
								pixel_in_cur_stream * sizeof(unsigned char),
								cudaMemcpyDeviceToHost,
								streams[i]),
								"Host memory asynchronized copy failed!");
	}
    
	// synchronize
    //cudaDeviceSynchronize();

    //拷贝输出图像数据至主机，并写入到本地
    cudaMemcpy(dst_img.data, d_pixels_out, pixel_cnt *  sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    imwrite("../stream.jpg", dst_img);
	cudaEventRecord(stop);
	// cudaEventSynchronize(): wait until the completion of all device work preceding the most recent call to cudaEventRecored()
	
	
	std::cout<<"GPU time: "<<milliseconds<< " ms"<<std::endl;

	// destroy streams
	for( int i=0; i<STREAMS_CNT; i++ ){
		cudaStreamDestroy(streams[i]);
	}
	// release the memory in device
	SAFE_CALL(cudaFree(d_pixels_in),"Free device color memory failed!");
    SAFE_CALL(cudaFree(d_pixels_out), "Free device gray memory failed!");
    return 0;
}
