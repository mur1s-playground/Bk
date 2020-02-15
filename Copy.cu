#include "Copy.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

__global__ void copyKernel(const unsigned int* device_data_input,
    const unsigned int input_position, const unsigned int width, const unsigned int height, const unsigned int channels,
    unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int output_start_x1, const unsigned int output_start_y1, const unsigned int alpha, const unsigned int type
    ) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height * channels) {
        int current_channel = i / (width * height);
        int current_idx = i % (width * height);
        int current_x = (current_idx % width);
        int current_y = (current_idx / width);

        if (type == 0) {
            unsigned char* input = (unsigned char*)&device_data_input[input_position];
            unsigned char* output = (unsigned char*)&device_data_output[output_position];

            if (current_channel >= channels && alpha == 255) {
                output[(output_start_y1 + current_y) * (output_width * output_channels) + (output_start_x1 + current_x) * output_channels + current_channel] = 0;
            }
            else {
                if (alpha < 255) {
                    output[(output_start_y1 + current_y) * (output_width * output_channels) + (output_start_x1 + current_x) * output_channels + current_channel] = (unsigned char)((255 - alpha) / 255.0f * output[(output_start_y1 + current_y) * (output_width * output_channels) + (output_start_x1 + current_x) * output_channels + current_channel] + (alpha / 255.0f) * input[current_y * (width * channels) + current_x * channels + current_channel]) / 2;
                } else {
                    output[(output_start_y1 + current_y) * (output_width * output_channels) + (output_start_x1 + current_x) * output_channels + current_channel] = input[current_y * (width * channels) + current_x * channels + current_channel];
                }
            }
        } else {
            float* input = (float*)&device_data_input[input_position];
            float* output = (float*)&device_data_output[output_position];

            if (current_channel >= channels && alpha == 255) {
                output[(output_start_y1 + current_y) * (output_width * output_channels) + (output_start_x1 + current_x) * output_channels + current_channel] = 0;
            } else {
                if (alpha < 255) {
                    output[(output_start_y1 + current_y) * (output_width * output_channels) + (output_start_x1 + current_x) * output_channels + current_channel] = ((255 - alpha) / 255.0f * output[(output_start_y1 + current_y) * (output_width * output_channels) + (output_start_x1 + current_x) * output_channels + current_channel] + (alpha / 255.0f) * input[current_y * (width * channels) + current_x * channels + current_channel]) / 2;
                } else {
                    output[(output_start_y1 + current_y) * (output_width * output_channels) + (output_start_x1 + current_x) * output_channels + current_channel] = input[current_y * (width * channels) + current_x * channels + current_channel];
                }
            }
        }
    }
}

void launch_copy(const unsigned int* device_data_input,
    const unsigned int input_position, const unsigned int width, const unsigned int height, const unsigned int channels,
    unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int output_start_x1, const unsigned int output_start_y1, const unsigned int alpha, const unsigned int type
    ) {
    
    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height * output_channels + threadsPerBlock - 1) / threadsPerBlock;

    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(device_data_input, input_position, width, height, channels, device_data_output, output_position, output_width, output_height, output_channels, output_start_x1, output_start_y1, alpha, type);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in copyKernel (error code %s)\n", cudaGetErrorString(err));
    }
}