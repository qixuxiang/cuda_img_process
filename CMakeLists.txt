cmake_minimum_required(VERSION 2.8)

project(cuda_img_process)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

set(CUDA_NVCC_FLAGS
    #-gencode arch=compute_60,code=sm_60;
    #-gencode arch=compute_61,code=sm_61;
    -gencode arch=compute_62,code=sm_62;
    -gencode arch=compute_62,code=compute_62;
    )

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr; -std=c++11)


set(SOURCE_FILES gray.cu)

cuda_add_executable(main ${SOURCE_FILES})



target_link_libraries(main ${OpenCV_LIBS} cublas cudnn cublas cudart)