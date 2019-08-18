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
  
__global__ void addKernel(uchar **pSrcImg,  uchar* pDstImg, int imgW, int imgH)  
{  
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;  
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;  
    if (tidx<imgW && tidy<imgH)
    {
        int idx=tidy*imgW+tidx;
        uchar lenaValue=pSrcImg[0][idx];
        uchar moonValue=pSrcImg[1][idx];
        pDstImg[idx]= uchar(0.5*lenaValue+0.5*moonValue);
    }
}  

int main()  
{  
    //OpenCV读取两幅图像
    Mat img[2];
    img[0]=imread("data/222.jpg", 0);
    img[1]=imread("data/333.jpg", 0);
    int imgH=img[0].rows;
    int imgW=img[0].cols;
    //输出图像
    Mat dstImg=Mat::zeros(imgH, imgW, CV_8UC1);
    //主机指针
    uchar **pImg=(uchar**)malloc(sizeof(uchar*)*2); //输入 二级指针

    //设备指针
    uchar **pDevice;//输入 二级指针
    uchar *pDeviceData;//输入 一级指针
    uchar *pDstImgData;//输出图像对应设备指针

    //分配GPU内存
    //目标输出图像分配GPU内存
    cudaMalloc(&pDstImgData, imgW*imgH*sizeof(uchar));
    //设备二级指针分配GPU内存
    cudaMalloc(&pDevice, sizeof(uchar*)*2);
    //设备一级指针分配GPU内存
    cudaMalloc(&pDeviceData, sizeof(uchar)*imgH*imgW*2);
    
    //关键：主机二级指针指向设备一级指针位置，这样才能使设备的二级指针指向设备的一级指针位置
    for (int i=0; i<2; i++)
    {
        pImg[i]=pDeviceData+i*imgW*imgH;
    }

    //拷贝数据到GPU
    //拷贝主机二级指针中的元素到设备二级指针指向的GPU位置 （这个二级指针中的元素是设备中一级指针的地址）
    cudaMemcpy(pDevice, pImg, sizeof(uchar*)*2, cudaMemcpyHostToDevice);
    //拷贝图像数据(主机一级指针指向主机内存) 到 设备一级指针指向的GPU内存中
    cudaMemcpy(pDeviceData, img[0].data, sizeof(uchar)*imgH*imgW, cudaMemcpyHostToDevice);
    cudaMemcpy(pDeviceData+imgH*imgW, img[1].data, sizeof(uchar)*imgH*imgW, cudaMemcpyHostToDevice);

    //核函数实现lena图和moon图的简单加权和
    dim3 block(8, 8);
    dim3 grid( (imgW+block.x-1)/block.x, (imgH+block.y-1)/block.y);
    addKernel<<<grid, block>>>(pDevice, pDstImgData, imgW, imgH);
    cudaThreadSynchronize();

    //拷贝输出图像数据至主机，并写入到本地
    cudaMemcpy(dstImg.data, pDstImgData, imgW*imgH*sizeof(uchar), cudaMemcpyDeviceToHost);
    imwrite("Thsis.jpg", dstImg);
}  