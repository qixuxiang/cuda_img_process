cmake_minimum_required(VERSION 2.8)
project(cuda_img_process)
set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

cuda_add_executable(combine combine.cu)
cuda_add_executable(gray gray.cu)


target_link_libraries(combine ${OpenCV_LIBS})
target_link_libraries(gray ${OpenCV_LIBS})
