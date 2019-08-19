#include "common.h"
#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

struct point{
    double x;
    double y;
};

__global__ void testFunc(point* d_a)
{
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {  
       d_a->x=100.0;  
       d_a->y = 100.0;    
    }
}

int main()
{ 
    point *cpu_pt, *gpu_pt;
    cpu_pt = (point*)malloc(sizeof(point));
    cpu_pt->x = 10;
    cpu_pt->y = 10;
    cudaMalloc((void**)&gpu_pt, sizeof(point));
    cudaMemcpy(gpu_pt, cpu_pt, sizeof(point), cudaMemcpyHostToDevice);
    dim3 dimblock(16, 16);
    dim3 dimgrid(1, 1);
    testFunc<<<dimgrid, dimblock>>>(gpu_pt);
    cudaMemcpy(cpu_pt, gpu_pt,sizeof(point),cudaMemcpyDeviceToHost); 
    printf("cpu_pt->x is %lf, cpu_pt->y is %lf\n", cpu_pt->x,cpu_pt->y);
    return 0;

}