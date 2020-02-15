#ifndef DRAWPLAYERS_HPP
#define DRAWPLAYERS_HPP

void launch_draw_players_kernel(const unsigned int* device_data_assets, const unsigned int players_models_position,
    const unsigned int* device_data_players, const unsigned int players_position,
    unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter);

#endif

