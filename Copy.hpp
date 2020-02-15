#ifndef COPY_HPP
#define COPY_HPP

void launch_copy(const unsigned int* device_data_input,
    const unsigned int input_position, const unsigned int width, const unsigned int height, const unsigned int channels,
    unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int output_start_x1, const unsigned int output_start_y1, const unsigned int alpha, const unsigned int type
);

#endif