#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct seq_s seq_t;
struct seq_s {
    int len;
    int *raw;
};

__global__ void kernel4(seq_s *gpu_seq)
{
	int idx = threadIdx.x;
	printf("idx:%d gpu_seq->len:%d \n", idx, gpu_seq[idx].len);
	printf("idx:%d gpu_seq->raw:%d \n",idx, *gpu_seq[idx].raw);

	gpu_seq[idx].len = idx*idx;
	*gpu_seq[idx].raw = 10*idx;

}

void test4();
int main() {
	  test4();
}

//传递结构体数组 结构体中包含 指针和非指针
void test4(){

	seq_s ** seqs;
	seqs = (seq_s **)malloc(5 * sizeof(seq_s) );
	for (size_t i = 0; i < 5; i++)
	{
		seq_s * seq;
		seq = (seq_s *)malloc(sizeof(seq_s));
		seq->raw = (int *)malloc(sizeof(int));

		*seq->raw = 5*i;
		seq->len = i;
		seqs[i] = seq;
	}

    for (size_t i = 0; i < 5; i++)
    {
        printf("before seqs[%d]->len:%d \n", i, seqs[i]->len);
        printf("before seqs[%d]->raw:%d \n", i, *seqs[i]->raw);
    }

 //这里定义一个用来存储的中间变量,这里存储的指针是GPU的指针
	seq_s * tmp_seq = (seq_s *)malloc(5 * sizeof(seq_s));
	for (size_t i = 0; i < 5; i++)
	{
		memcpy(&tmp_seq[i], seqs[i],  sizeof(seq_s) );
	}



	for (size_t i = 0; i < 5; i++)
	{
		cudaMalloc(&(tmp_seq[i].raw),   sizeof(int));
		cudaMemcpy(tmp_seq[i].raw, seqs[i]->raw, sizeof(int), cudaMemcpyHostToDevice);

	}
	seq_s * gpu_seq;

	cudaMalloc((void**)&gpu_seq, 5 * sizeof(seq_s));

	cudaMemcpy(gpu_seq, tmp_seq, 5 * sizeof(seq_s), cudaMemcpyHostToDevice);

	kernel4 << <1, 5 >> >(gpu_seq);

//在把数据从GPU传递到CPU时,也还要用到 中间变量存储的GPU指针 这是因为gpu_seq是gpu指针我们无法直接获取里面的数据
	seq_s * new_seq = (seq_s *)malloc(5 * sizeof(seq_s));
	cudaMemcpy(new_seq, gpu_seq, 5 * sizeof(seq_s), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < 5; i++)
	{
		new_seq[i].raw= (int *)malloc(sizeof(int));
		cudaMemcpy(new_seq[i].raw, tmp_seq[i].raw, sizeof(int), cudaMemcpyDeviceToHost);
	}

	for (size_t i = 0; i < 5; i++)
	{
		printf("after new_seq[%d].len :%d \n",i, new_seq[i].len);
		printf("after new_seq[%d].raw :%d \n",i, *new_seq[i].raw);
	}

}

