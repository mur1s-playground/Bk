#include "Resize.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__forceinline__
__device__ float getInterpixel(const unsigned char* frame, const unsigned int width, const unsigned int height, const unsigned int channels, float x, float y, const int c) {
    int x_i = (int)x;
    int y_i = (int)y;
    x -= x_i;
    y -= y_i;
    
    unsigned char value_components[4];
    value_components[0] = frame[y_i * (width * channels) + x_i * channels + c];
    if (x > 0) {
        if (x_i + 1 < width) {
            value_components[1] = frame[y_i * (width * channels) + (x_i + 1) * channels + c];
        } else {
            x = 0.0f;
        }
    } 
    if (y > 0) {
        if (y_i + 1 < height) {
            value_components[2] = frame[(y_i + 1) * (width * channels) + x_i * channels + c];
            if (x > 0) {
                value_components[3] = frame[(y_i + 1) * (width * channels) + (x_i + 1) * channels + c];
            }
        } else {
            y = 0.0f;
        }
    }
    
    float m_0 = 4.0f / 16.0f;
    float m_1 = 4.0f / 16.0f;
    float m_2 = 4.0f / 16.0f;
    float m_3 = 4.0f / 16.0f;
    float tmp, tmp2;
    if (x <= 0.5f) {
        tmp = ((0.5f - x) / 0.5f) * m_1;
        m_0 += tmp;
        m_1 -= tmp;
        m_2 += tmp;
        m_3 -= tmp;
    } else {
        tmp = ((x - 0.5f) / 0.5f) * m_0;
        m_0 -= tmp;
        m_1 += tmp;
        m_2 -= tmp;
        m_3 += tmp;
    }
    if (y <= 0.5f) {
        tmp = ((0.5f - y) / 0.5f) * m_2;
        tmp2 = ((0.5f - y) / 0.5f) * m_3;
        m_0 += tmp;
        m_1 += tmp2;
        m_2 -= tmp;
        m_3 -= tmp2;
    } else {
        tmp = ((y - 0.5f) / 0.5f) * m_0;
        tmp2 = ((y - 0.5f) / 0.5f) * m_1;
        m_0 -= tmp;
        m_1 -= tmp2;
        m_2 += tmp;
        m_3 += tmp2;
    }
    float value = m_0 * value_components[0] + m_1 * value_components[1] + m_2 * value_components[2] + m_3 * value_components[3];
    return value;
}

__global__ void resizeKernel(const unsigned int* device_data,
    const unsigned int frame_position,
        const unsigned int width, const unsigned int height, const unsigned int channels,
        const unsigned int crop_x1, const unsigned int crop_x2, const unsigned int crop_y1, const unsigned int crop_y2,
        unsigned int *device_data_output, const unsigned int frame_position_target,
        const unsigned int width_target, const unsigned int height_target,
    const float sampling_filter_width_ratio, const unsigned int sampling_filter_width, const float sampling_filter_height_ratio, const unsigned int sampling_filter_height
    ) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < width_target * height_target) {
        int current_x = (i % width_target);
        int current_y = (i / width_target);

        unsigned char* frame = (unsigned char*)&device_data[frame_position];
        unsigned char* target_frame = (unsigned char*)&device_data_output[frame_position_target];
        
        float current_source_x = crop_x1 + (current_x*sampling_filter_width_ratio);
        float current_source_y = crop_y1 + (current_y*sampling_filter_height_ratio);

        int current_source_x_i = (int)floorf(current_source_x);
        int current_source_y_i = (int)floorf(current_source_y);

        float components[3];
        float value[3];
        for (int c = 0; c < channels; c++) {
            components[c] = 0.0f;
            value[c] = 0.0f;
        }
        for (int y = 0; y < sampling_filter_height; y++) {
            for (int x = 0; x < sampling_filter_width; x++) {
                if (current_source_y_i+y < height && current_source_x_i + x < width) {
                    for (int c = 0; c < channels; c++) {
                        value[c] += getInterpixel(frame, width, height, channels, current_source_x+x, current_source_y+y, c);
                        components[c] += 1.0f;
                    }
                }
            }
        }
        for (int c = 0; c < channels; c++) {
            target_frame[current_y * (width_target * channels) + current_x * channels + c] = (unsigned char) roundf(value[c] / components[c]);
        }
    }
}

void launch_resize(const unsigned int* device_data,
    const unsigned int frame_position,
    const unsigned int width, const unsigned int height, const unsigned int channels,
    const unsigned int crop_x1, const unsigned int crop_x2, const unsigned int crop_y1, const unsigned int crop_y2,
    unsigned int* device_data_output, const unsigned int frame_position_target,
    const unsigned int width_target, const unsigned int height_target) {

    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (width_target * height_target + threadsPerBlock - 1) / threadsPerBlock;

    float sampling_filter_width_ratio = (crop_x2 - crop_x1) / (float)(width_target);
    int sampling_filter_width = (int)ceilf(sampling_filter_width_ratio);
    float sampling_filter_height_ratio = (crop_y2 - crop_y1) / (float)(height_target);
    int sampling_filter_height = (int)ceilf(sampling_filter_height_ratio);
    
    resizeKernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, frame_position, width, height, channels, crop_x1, crop_x2, crop_y1, crop_y2, device_data_output, frame_position_target, width_target, height_target, sampling_filter_width_ratio, sampling_filter_width, sampling_filter_height_ratio, sampling_filter_height);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in resizeKernel (error code %s)\n", cudaGetErrorString(err));
    }
}