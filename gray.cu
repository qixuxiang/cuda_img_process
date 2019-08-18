#include "common.h"
#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;//包含cv命名空间

#define CHANNLES 3

__global__ void grayScale(uchar3 const *rgb ,unsigned char *gray, int width, int height)
{
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if(x < width && y < height)
        {
            //获取灰度图的全局坐标，即矩阵拉伸为一维数组
            int offset = y * width + x;  
            // 分别获取RGB三个通道对应的数值
            uchar3 pixel = rgb[offset];
            //利用公式计算灰度值并赋值
            gray[offset] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
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
    //设备指针 device
   
    unsigned char *deviceInputGray;
    uchar3 *deviceInputRGB;
    cudaMalloc((void **) &deviceInputGray, imgH * imgW * sizeof(unsigned char));
    cudaMalloc((void **) &deviceInputRGB, imgH * imgW  * sizeof(uchar3));
    //cudaMemcpy(deviceInputGray, dst_img.data, imgW * imgH * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInputRGB, src_img.data, imgW * imgH * sizeof(uchar3), cudaMemcpyHostToDevice);
    

    //核函数实现图的灰度化
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgW + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    grayScale <<<blocksPerGrid, threadsPerBlock>>> (deviceInputRGB, deviceInputGray, imgW, imgH);
    cudaThreadSynchronize();
    //拷贝输出图像数据至主机，并写入到本地
    cudaMemcpy(dst_img.data, deviceInputGray, imgW * imgH *  sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //cudaMemcpy(grayImage.data, d_out, imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    imwrite("../Thsis.jpg", dst_img);
    CHECK(cudaDeviceReset());
    return 0;
}