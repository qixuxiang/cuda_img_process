#include "common.h"
#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;
  
__global__ void addKernel(uchar3 **pSrcImg,  uchar3* pDstImg, int imgW, int imgH)  
{  
    int x = threadIdx.x + blockDim.x * blockIdx.x;  
    int y = threadIdx.y + blockDim.y * blockIdx.y;  
    if (x < imgW && y < imgH)
    {
        //获取灰度图的全局坐标，即矩阵拉伸为一维数组
        int offset = y * imgW + x;  
        // 分别获取两张RGB三个通道对应的数值
        uchar3 pixel1 = pSrcImg[0][offset];
        uchar3 pixel2 = pSrcImg[1][offset];
        pDstImg[offset].x =uchar(pixel1.x + pixel2.x);
        pDstImg[offset].y =uchar(pixel1.y + pixel2.y);
        pDstImg[offset].z =uchar(pixel1.z + pixel2.z);
    }
}  

int main()  
{  
    //OpenCV读取两幅图像
    Mat img[2];
    img[0]=imread("../data/test.jpg");
    img[1]=imread("../data/NASA_Mars_Rover.jpg");
    int imgH=img[0].rows;
    int imgW=img[0].cols;
    //输出图像
    Mat dstImg=Mat::zeros(imgH, imgW, CV_8UC3);
    //主机指针
    uchar3 **pImg=(uchar3**)malloc(sizeof(uchar3*)*2); //输入 二级指针

    //设备指针
    uchar3 **pDevice;//输入 二级指针
    uchar3 *pDeviceData;//输入 一级指针
    uchar3 *pDstImgData;//输出图像对应设备指针

    //分配GPU内存
    //目标输出图像分配GPU内存
    cudaMalloc(&pDstImgData, imgW*imgH*sizeof(uchar3));
    //设备二级指针分配GPU内存
    cudaMalloc(&pDevice, sizeof(uchar3*)*2);
    //设备一级指针分配GPU内存
    cudaMalloc(&pDeviceData, sizeof(uchar3)*imgH*imgW*2);
    
    //关键：主机二级指针指向设备一级指针位置，这样才能使设备的二级指针指向设备的一级指针位置
    for (int i=0; i<2; i++)
    {
        pImg[i]=pDeviceData+i*imgW*imgH;
    }

    //拷贝数据到GPU
    //拷贝主机二级指针中的元素到设备二级指针指向的GPU位置 （这个二级指针中的元素是设备中一级指针的地址）
    cudaMemcpy(pDevice, pImg, sizeof(uchar3*)*2, cudaMemcpyHostToDevice);
    //拷贝图像数据(主机一级指针指向主机内存) 到 设备一级指针指向的GPU内存中
    cudaMemcpy(pDeviceData, img[0].data, sizeof(uchar3)*imgH*imgW, cudaMemcpyHostToDevice);
    cudaMemcpy(pDeviceData+imgH*imgW, img[1].data, sizeof(uchar3)*imgH*imgW, cudaMemcpyHostToDevice);

    //核函数实现lena图和moon图的简单加权和
    dim3 block(8, 8);
    dim3 grid( (imgW+block.x-1)/block.x, (imgH+block.y-1)/block.y);
    addKernel<<<grid, block>>>(pDevice, pDstImgData, imgW, imgH);
    cudaThreadSynchronize();

    //拷贝输出图像数据至主机，并写入到本地
    cudaMemcpy(dstImg.data, pDstImgData, imgW*imgH*sizeof(uchar3), cudaMemcpyDeviceToHost);
    imwrite("../Thsis.jpg", dstImg);
    CHECK(cudaDeviceReset());
    return 0;
}  