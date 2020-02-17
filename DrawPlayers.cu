#include "DrawPlayers.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Player.hpp"
#include "Grid.hpp"

#include "stdio.h"

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
        }
        else {
            x = 0.0f;
        }
    }
    if (y > 0) {
        if (y_i + 1 < height) {
            value_components[2] = frame[(y_i + 1) * (width * channels) + x_i * channels + c];
            if (x > 0) {
                value_components[3] = frame[(y_i + 1) * (width * channels) + (x_i + 1) * channels + c];
            }
        }
        else {
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
    }
    else {
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
    }
    else {
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

__global__ void draw_players_kernel(
        const unsigned int* device_data_assets, const unsigned int players_models_position,
        const unsigned int* device_data_players, const unsigned int players_position, const unsigned int gd_position_in_bf, const unsigned int gd_data_position_in_bf,
        unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
        const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int players_count = device_data_players[players_position-1] / (unsigned int)ceilf(sizeof(struct player) / (float)sizeof(unsigned int));
    struct player* players = (struct player*) &device_data_players[players_position];
    struct player_model* player_models = (struct player_model*) &device_data_assets[players_models_position];

    if (i < output_width * output_height * output_channels) {
        int current_channel = i / (output_width * output_height);
        int current_idx = i % (output_width * output_height);
        int current_x = (current_idx % output_width);
        int current_y = (current_idx / output_width);

        float current_game_x = camera_x1 + current_x*camera_z;
        float current_game_y = camera_y1 + current_y*camera_z;

        int sampling_filter_dim = ceilf(camera_z);

        

        unsigned char* output = (unsigned char*)&device_data_output[output_position];
        /*
        if ((int)(current_game_x) % 32 == 0 || (int)(current_game_y) % 32 == 0) {
            output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 255;
            output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255;
            output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255;
            output[current_y * (output_width * output_channels) + current_x * output_channels + 3] = 255;
        }
        */
        int grid_current_idx = grid_get_index(device_data_players, gd_position_in_bf, struct vector3<float> (current_game_x, current_game_y, 0.0f));
        if (grid_current_idx != -1) {
            unsigned int entities_iddata_position = device_data_players[gd_data_position_in_bf + 1 + grid_current_idx];
            if (entities_iddata_position > 0) {
                unsigned int entities_count = device_data_players[entities_iddata_position];

                for (int e = 0; e < entities_count; e++) {
                    unsigned int entity_id = device_data_players[entities_iddata_position + 1 + e];
                    if (entity_id < UINT_MAX) {
                        /*
                        output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 0;
                        output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255 * (e+1 % 2);
                        output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255 * (e % 2);
                        output[current_y * (output_width * output_channels) + current_x * output_channels + 3] = 100;
                        */
                        struct player_model* pm = &player_models[players[entity_id].model_id];

                        unsigned int shadow_positions = pm->shadow_positions;
                        float down_scale = 1;
                        if (camera_z / pm->shadow_scale > 2) {
                            shadow_positions = pm->shadow_med_positions;
                            down_scale *= 2;
                        }
                        if (camera_z / (pm->shadow_scale * down_scale) > 2) {
                            shadow_positions = pm->shadow_lo_positions;
                            down_scale *= 2;
                        }
                        sampling_filter_dim = ceilf(camera_z / (pm->shadow_scale * down_scale));

                        float offset_to_player_shadow_base_x = (current_game_x - (players[entity_id].position[0] + pm->shadow_offset[0])) / (pm->shadow_scale * down_scale);
                        float offset_to_player_shadow_base_y = (current_game_y - (players[entity_id].position[1] + pm->shadow_offset[1])) / (pm->shadow_scale * down_scale);

                        //printf("sfd %d, position_x %f, position_y %f, game_x %f, game_y %f, grid_current_idx %d, entities_iddata_pos: %d, entities_count %d, entities_id %d, oofset_x %f, offset_y %f, down_scale %f\n\n", sampling_filter_dim, players[entity_id].position[0], players[entity_id].position[1], current_game_x, current_game_y, grid_current_idx, entities_iddata_position, entities_count, entity_id, offset_to_player_shadow_base_x, offset_to_player_shadow_base_y, down_scale);

                        if (offset_to_player_shadow_base_x >= 1 && offset_to_player_shadow_base_x < pm->shadow_dimensions[0] / down_scale - 1 &&
                            offset_to_player_shadow_base_y >= 1 && offset_to_player_shadow_base_y < pm->shadow_dimensions[1] / down_scale - 1) {

                            unsigned int* p_shadow_positions = (unsigned int*)&device_data_assets[shadow_positions];
                            unsigned char* p_shadow = (unsigned char*)&device_data_assets[p_shadow_positions[(int)(players[entity_id].orientation / 10) % 36]];

                            //enum player_stance p_stance = players[p].player_stance;
                            //enum player_action p_action = players[p].player_action;

                            for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
                                for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
                                    if (offset_to_player_shadow_base_x + s_x >= 1 && offset_to_player_shadow_base_x + s_x < pm->shadow_dimensions[0] / down_scale - 1 &&
                                        offset_to_player_shadow_base_y + s_y >= 1 && offset_to_player_shadow_base_y + s_y < pm->shadow_dimensions[1] / down_scale - 1
                                        ) {

                                        float model_palette_idx_x = offset_to_player_shadow_base_x + s_x;
                                        float model_palette_idx_y = offset_to_player_shadow_base_y + s_y;

                                        float interpixel_alpha = getInterpixel(p_shadow, pm->shadow_dimensions[0] / down_scale, pm->shadow_dimensions[1] / down_scale, 4, model_palette_idx_x, model_palette_idx_y, 3);
                                        if (interpixel_alpha > 25) {
                                            float interpixel = getInterpixel(p_shadow, pm->shadow_dimensions[0] / down_scale, pm->shadow_dimensions[1] / down_scale, 4, model_palette_idx_x, model_palette_idx_y, current_channel);
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (interpixel_alpha / 255.0f) * interpixel));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                /*
                float player_y_max = -1.0f;
                int player_id_max = -1;
                for (int e = 0; e < entities_count; e++) {
                    unsigned int entity_id = device_data_players[entities_iddata_position + 1 + e];
                    if (entity_id < UINT_MAX) {
                        struct player_model* pm = &player_models[players[entity_id].model_id];
                        unsigned int model_positions = pm->model_positions;
                        float down_scale = 1;
                        if (camera_z / pm->model_scale > 2) {
                            model_positions = pm->model_med_positions;
                            down_scale *= 2;
                        }
                        if (camera_z / (pm->model_scale * down_scale) > 2) {
                            model_positions = pm->model_lo_positions;
                            down_scale *= 2;
                        }

                        sampling_filter_dim = ceilf(camera_z / (pm->model_scale * down_scale));

                        float offset_to_player_base_x = (current_game_x - (players[entity_id].position[0])) / (pm->model_scale * down_scale);
                        float offset_to_player_base_y = (current_game_y - (players[entity_id].position[1])) / (pm->model_scale * down_scale);

                        if (offset_to_player_base_x >= 1 && offset_to_player_base_x < pm->model_dimensions[0] / down_scale - 1 &&
                            offset_to_player_base_y >= 1 && offset_to_player_base_y < pm->model_dimensions[1] / down_scale - 1) {

                            unsigned int* p_model_positions = (unsigned int*)&device_data_assets[model_positions];
                            unsigned char* p_model = (unsigned char*)&device_data_assets[p_model_positions[(int)(players[entity_id].orientation / 10) % 36]];

                            for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
                                for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
                                    if (offset_to_player_base_x + s_x >= 1 && offset_to_player_base_x + s_x < pm->model_dimensions[0] / down_scale - 1 &&
                                        offset_to_player_base_y + s_y >= 1 && offset_to_player_base_y + s_y < pm->model_dimensions[1] / down_scale - 1
                                        ) {
                                        float model_palette_idx_x = offset_to_player_base_x + s_x;
                                        float model_palette_idx_y = offset_to_player_base_y + s_y;

                                        float interpixel_alpha = getInterpixel(p_model, pm->model_dimensions[0] / down_scale, pm->model_dimensions[1] / down_scale, 4, model_palette_idx_x, model_palette_idx_y, 3);
                                        if (interpixel_alpha >= 128) {
                                            if (players[entity_id].position[1] > player_y_max) {
                                                player_y_max = players[entity_id].position[1];
                                                player_id_max = entity_id;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                */
                //if (player_id_max >= 0) {
                for (int e = 0; e < entities_count; e++) {
                    unsigned int entity_id = device_data_players[entities_iddata_position + 1 + e];
                    if (entity_id < UINT_MAX) {
                        struct player_model* pm = &player_models[players[entity_id].model_id];
                        unsigned int model_positions = pm->model_positions;
                        float down_scale = 1;
                        if (camera_z / pm->model_scale > 2) {
                            model_positions = pm->model_med_positions;
                            down_scale *= 2;
                        }
                        if (camera_z / (pm->model_scale * down_scale) > 2) {
                            model_positions = pm->model_lo_positions;
                            down_scale *= 2;
                        }

                        sampling_filter_dim = ceilf(camera_z / (pm->model_scale * down_scale));

                        float offset_to_player_base_x = (current_game_x - (players[entity_id].position[0])) / (pm->model_scale * down_scale);
                        float offset_to_player_base_y = (current_game_y - (players[entity_id].position[1])) / (pm->model_scale * down_scale);

                        if (offset_to_player_base_x >= 1 && offset_to_player_base_x < pm->model_dimensions[0] / down_scale - 1 &&
                            offset_to_player_base_y >= 1 && offset_to_player_base_y < pm->model_dimensions[1] / down_scale - 1) {

                            unsigned int* p_model_positions = (unsigned int*)&device_data_assets[model_positions];
                            unsigned char* p_model = (unsigned char*)&device_data_assets[p_model_positions[(int)(players[entity_id].orientation / 10) % 36]];

                            //enum player_stance p_stance = players[p].player_stance;
                            //enum player_action p_action = players[p].player_action;

                            for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
                                for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
                                    if (offset_to_player_base_x + s_x >= 1 && offset_to_player_base_x + s_x < pm->model_dimensions[0] / down_scale - 1 &&
                                        offset_to_player_base_y + s_y >= 1 && offset_to_player_base_y + s_y < pm->model_dimensions[1] / down_scale - 1
                                        ) {

                                        float model_palette_idx_x = offset_to_player_base_x + s_x;
                                        float model_palette_idx_y = offset_to_player_base_y + s_y;

                                        float interpixel_alpha = getInterpixel(p_model, pm->model_dimensions[0] / down_scale, pm->model_dimensions[1] / down_scale, 4, model_palette_idx_x, model_palette_idx_y, 3);
                                        if (interpixel_alpha > 0) {
                                            float interpixel = getInterpixel(p_model, pm->model_dimensions[0] / down_scale, pm->model_dimensions[1] / down_scale, 4, model_palette_idx_x, model_palette_idx_y, current_channel);
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (interpixel_alpha / 255.0f) * interpixel));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void launch_draw_players_kernel(const unsigned int* device_data_assets, const unsigned int players_models_position,
    const unsigned int* device_data_players, const unsigned int players_position, const unsigned int gd_position_in_bf, const unsigned int gd_data_position_in_bf,
    unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter) {

    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_width * output_height * 3 + threadsPerBlock - 1) / threadsPerBlock;

    draw_players_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data_assets, players_models_position, device_data_players, players_position, gd_position_in_bf, gd_data_position_in_bf, device_data_output, output_position, output_width, output_height, output_channels, camera_x1, camera_y1, camera_z, tick_counter);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in draw_players_kernel (error code %s)\n", cudaGetErrorString(err));
    }
}