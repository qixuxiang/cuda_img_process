cmake_minimum_required(VERSION 2.8)
project(cuda_img_process)
set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

cuda_add_executable(struct_test struct_pointer.cu)
cuda_add_executable(struct_array struct_array.cu)
cuda_add_executable(combine_gray combine_gray.cu)
cuda_add_executable(combine_rgb combine_rgb.cu)
cuda_add_executable(gray gray.cu)
cuda_add_executable(multi_stream_gray multi_stream_gray.cu)

target_link_libraries(combine_gray ${OpenCV_LIBS})
target_link_libraries(combine_rgb ${OpenCV_LIBS})
target_link_libraries(gray ${OpenCV_LIBS})
target_link_libraries(multi_stream_gray ${OpenCV_LIBS})
