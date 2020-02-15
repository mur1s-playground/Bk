#ifndef RESIZE_HPP
#define RESIZE_HPP

void launch_resize(const unsigned int* device_data,
    const unsigned int frame_position,
    const unsigned int width, const unsigned int height, const unsigned int channels,
    const unsigned int crop_x1, const unsigned int crop_x2, const unsigned int crop_y1, const unsigned int crop_y2,
    unsigned int* device_data_output, const unsigned int frame_position_target,
    const unsigned int width_target, const unsigned int height_target);

#endif