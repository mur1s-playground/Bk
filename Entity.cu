#include "Entity.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Vector2.hpp"
#include "Grid.hpp"
#include "Entity.hpp"
#include "Player.hpp"
#include "Item.hpp"
#include "Map.hpp"
#include "Model.hpp"

using namespace std;

unsigned int                entities_size_in_bf = 0;
unsigned int				entities_position = 0;
vector<struct entity>       entities;

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

__global__ void draw_entities_kernel(
        const unsigned int* device_data_assets, const unsigned int players_models_position, const unsigned int item_models_position, const unsigned int map_models_position, const unsigned int font_position,
        const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int gd_position_in_bf, const unsigned int gd_data_position_in_bf,
        unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
        const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const struct vector2<unsigned int> mouse_position, const unsigned int tick_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //unsigned int players_count = device_data_players[players_position-1] / (unsigned int)ceilf(sizeof(struct player) / (float)sizeof(unsigned int));
    //struct player* players = (struct player*) &device_data_players[players_position];

    struct entity* entities = (struct entity*) &device_data_rw[entities_position];
    
    struct model* player_models = (struct model*) &device_data_assets[players_models_position];
    struct model* item_models = (struct model*) &device_data_assets[item_models_position];
    struct model* map_models = (struct model*) &device_data_assets[map_models_position];

    if (i < output_width * output_height * output_channels) {
        int current_channel = i / (output_width * output_height);
        int current_idx = i % (output_width * output_height);
        int current_x = (current_idx % output_width);
        int current_y = (current_idx / output_width);

        float current_game_x = camera_x1 + current_x*camera_z;
        float current_game_y = camera_y1 + current_y*camera_z;

        float current_mouse_game_x = camera_x1 + mouse_position[0] * camera_z;
        float current_mouse_game_y = camera_y1 + mouse_position[1] * camera_z;

        int sampling_filter_dim = ceilf(camera_z);



        unsigned char* output = (unsigned char*)&device_data_output[output_position];

        /*if ((int)(current_game_x) % 32 == 0 || (int)(current_game_y) % 32 == 0) {
            output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 255;
            output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255;
            output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255;
            output[current_y * (output_width * output_channels) + current_x * output_channels + 3] = 255;
        }*/
        
        int grid_current_idx = grid_get_index(device_data_rw, gd_position_in_bf, struct vector3<float> (current_game_x, current_game_y, 0.0f));
        if (grid_current_idx != -1) {
            unsigned int entities_iddata_position = device_data_rw[gd_data_position_in_bf + 1 + grid_current_idx];
            if (entities_iddata_position > 0) {
                unsigned int entities_count = device_data_rw[entities_iddata_position];

                for (int e = 0; e < entities_count; e++) {
                    unsigned int entity_id = device_data_rw[entities_iddata_position + 1 + e];
                    if (entity_id < UINT_MAX) {
                        /*
                        if ((int)(current_game_x) % 32 == 0 || (int)(current_game_y) % 32 == 0) {
                            output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 0;
                            output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255 * (e + 1 % 2);
                            output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255 * (e % 2);
                            output[current_y * (output_width * output_channels) + current_x * output_channels + 3] = 100;
                        }
                        */
                        /*
                        output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 0;
                        output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255 * (e+1 % 2);
                        output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255 * (e % 2);
                        output[current_y * (output_width * output_channels) + current_x * output_channels + 3] = 100;
                        */
                        struct model* m = nullptr;
                        if (entities[entity_id].et == ET_PLAYER) {
                            m = &player_models[entities[entity_id].model_id];
                        } else if (entities[entity_id].et == ET_ITEM) {
                            m = &item_models[entities[entity_id].model_id - 50];
                        } else if (entities[entity_id].et == ET_STATIC_ASSET) {
                            m = &map_models[entities[entity_id].model_id - 100];
                        }
                        
                        if (m != nullptr) {
                            const unsigned int* shadows_positions = &device_data_assets[m->shadow_positions];
                            int upscale = 1;
                            float upscale_fac = 1.0f;
                            unsigned int shadow_position = shadows_positions[upscale - 1];
                            while (camera_z / (m->shadow_scale / upscale_fac) < 2 && upscale - 1 < m->shadow_zoom_level_count - 1) {
                                upscale++;
                                shadow_position = shadows_positions[upscale - 1];
                                upscale_fac *= 2.0f;
                            }

                            sampling_filter_dim = ceilf(camera_z / (m->shadow_scale / upscale_fac));

                            float offset_to_model_shadow_base_x = (current_game_x - (entities[entity_id].position[0] + m->shadow_offset[0])) / (m->shadow_scale / upscale_fac);
                            float offset_to_model_shadow_base_y = (current_game_y - (entities[entity_id].position[1] + m->shadow_offset[1])) / (m->shadow_scale / upscale_fac);

                            if (offset_to_model_shadow_base_x >= 1 && offset_to_model_shadow_base_x < m->shadow_dimensions[0] * upscale_fac - 1 &&
                                offset_to_model_shadow_base_y >= 1 && offset_to_model_shadow_base_y < m->shadow_dimensions[1] * upscale_fac - 1) {
                                /*
                                output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 0;
                                output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255 * (e + 1 % 2);
                                output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255 * (e % 2);
                                output[current_y * (output_width * output_channels) + current_x * output_channels + 3] = 100;
                                */

                                unsigned int* p_shadow_positions = (unsigned int*)&device_data_assets[shadow_position];
                                unsigned char* p_shadow = (unsigned char*)&device_data_assets[p_shadow_positions[(int)(entities[entity_id].orientation / 10) % 36]];

                                //enum player_stance p_stance = players[p].player_stance;
                                //enum player_action p_action = players[p].player_action;

                                for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
                                    for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
                                        if (offset_to_model_shadow_base_x + s_x >= 1 && offset_to_model_shadow_base_x + s_x < m->shadow_dimensions[0] * upscale_fac - 1 &&
                                            offset_to_model_shadow_base_y + s_y >= 1 && offset_to_model_shadow_base_y + s_y < m->shadow_dimensions[1] * upscale_fac - 1
                                            ) {

                                            float model_palette_idx_x = offset_to_model_shadow_base_x + s_x;
                                            float model_palette_idx_y = offset_to_model_shadow_base_y + s_y;

                                            float interpixel_alpha = getInterpixel(p_shadow, m->shadow_dimensions[0] * upscale_fac, m->shadow_dimensions[1] * upscale_fac, 4, model_palette_idx_x, model_palette_idx_y, 3);
                                            if (interpixel_alpha > 25) {
                                                float interpixel = getInterpixel(p_shadow, m->shadow_dimensions[0] * upscale_fac, m->shadow_dimensions[1] * upscale_fac, 4, model_palette_idx_x, model_palette_idx_y, current_channel);
                                                output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (interpixel_alpha / 255.0f) * interpixel));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                float player_y_max = -1.0f;
                int player_id_max = -1;
                bool has_text = false;
                for (int e = 0; e < entities_count; e++) {
                    unsigned int entity_id = device_data_rw[entities_iddata_position + 1 + e];
                    if (entity_id < UINT_MAX) {
                        struct model* m = nullptr;
                        if (entities[entity_id].et == ET_PLAYER) {
                            m = &player_models[entities[entity_id].model_id];
                        } else if (entities[entity_id].et == ET_ITEM) {
                            m = &item_models[entities[entity_id].model_id - 50];
                        } else if (entities[entity_id].et == ET_STATIC_ASSET) {
                            m = &map_models[entities[entity_id].model_id - 100];
                        }

                        if (m != nullptr) {
                            const unsigned int* model_positions = &device_data_assets[m->model_positions];
                            int upscale = 1;
                            float upscale_fac = 1.0f;
                            unsigned int model_position = model_positions[upscale - 1];
                            while (camera_z / (m->model_scale / upscale_fac) < 2 && upscale - 1 < m->model_zoom_level_count - 1) {
                                upscale++;
                                model_position = model_positions[upscale - 1];
                                upscale_fac *= 2.0f;
                            }

                            sampling_filter_dim = ceilf(camera_z / (m->model_scale / upscale_fac));

                            float offset_to_model_base_x = (current_game_x - (entities[entity_id].position[0])) / (m->model_scale / upscale_fac);
                            float offset_to_model_base_y = (current_game_y - (entities[entity_id].position[1])) / (m->model_scale / upscale_fac);

                            if (entities[entity_id].et == ET_PLAYER) {
                                //inventory
                                int inventory_max_id = -1;
                                int* params = (int*)&entities[entity_id].params;
                                int params_pos = 1;
                                for (int ip = 0; ip < 6; ip++) {
                                    if (params[params_pos++] < UINT_MAX) inventory_max_id = ip;
                                    params_pos++;
                                }

                                if (offset_to_model_base_y < 32 * (inventory_max_id+1) && offset_to_model_base_y >= 0.0f && offset_to_model_base_x + 32.0f >= -3 - 18 && offset_to_model_base_x + 32.0f < 20) {
                                    output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = 200;

                                    if (offset_to_model_base_x + 32.0f + 19 >= 0 && offset_to_model_base_x + 32.0f + 19 < 32) {
                                        //inventory "text"
                                        int letter_idx = (int)(offset_to_model_base_y) / 32;
                                        int letter_code = -48;
                                        if (params[1 + letter_idx * 2] < UINT_MAX) {
                                            if (params[1 + letter_idx * 2] == 50) {
                                                letter_code += '[';
                                            }
                                            else if (params[1 + letter_idx * 2] == 51) {
                                                letter_code += ']';
                                            }
                                            else if (params[1 + letter_idx * 2] == 52) {
                                                letter_code += '`';
                                            }
                                            if (letter_code >= 0 && letter_code < 122 - 48) {
                                                unsigned char* letter = (unsigned char*)&device_data_assets[device_data_assets[font_position + letter_code]];
                                                int letter_x = (int)(offset_to_model_base_x + 32.0f + 19) % 32;
                                                int letter_y = (int)offset_to_model_base_y % 32;

                                                //shooting
                                                if (params[1 + letter_idx * 2] == 50 && params[1 + letter_idx * 2 + 1] % 15 != 0) {
                                                    if (letter_x >= 28 && letter_x <= 32 && letter_y >= 7 && letter_y <= 15) {
                                                        output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 255;
                                                        output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 0;
                                                        output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 0;
                                                    }
                                                }

                                                float letter_alpha = letter[letter_y * (32 * 4) + letter_x * 4 + 3];
                                                if (letter_alpha > 25) {
                                                    output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - letter_alpha) / 255.0f * 255 + (letter_alpha / 255.0f) * letter[letter_y * (32 * 4) + letter_x * 4 + current_channel]));
                                                }
                                            }
                                        }
                                    }
                                }

                                //top text bg
                                if (offset_to_model_base_y < 3 && offset_to_model_base_y >= -35.0f && offset_to_model_base_x + 32.0f >= -3 -18 && offset_to_model_base_x + 32.0f < entities[entity_id].name_len * 32 + 3) {
                                    int bg_alpha = 150;
                                    output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = 200;
                                }
                                //hp bar
                                if (offset_to_model_base_y < 2 && offset_to_model_base_y >= -33.0f && offset_to_model_base_x + 32.0f >= -19 && offset_to_model_base_x + 32.0f < -11) {
                                    float hp_percent = entities[entity_id].params[0] / (float)100.0f;
                                    float hp_scale_y = 31.0f;
                                    if ( hp_percent* hp_scale_y -33.0f >= offset_to_model_base_y) {
                                        if (hp_percent > 0.66f) {
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 0;
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255;
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 0;
                                        } else if (hp_percent > 0.33f) {
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 255;
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 157;
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 0;
                                        } else {
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 255;
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 0;
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 0;
                                        }
                                    }
                                }
                                //shield bar
                                if (offset_to_model_base_y < 2 && offset_to_model_base_y >= -33.0f && offset_to_model_base_x + 32.0f >= -11 && offset_to_model_base_x + 32.0f < -3) {
                                    float shield_percent = entities[entity_id].params[1] / (float)100.0f;
                                    float shield_scale_y = 31.0f;
                                    if (shield_percent * shield_scale_y - 33.0f >= offset_to_model_base_y) {
                                        output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 25;
                                        output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255;
                                        output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255;
                                    }
                                }
                                //top text
                                if (offset_to_model_base_y < 0 && offset_to_model_base_y >= -32.0f && offset_to_model_base_x + 32.0f >= 0 && offset_to_model_base_x + 32.0f < entities[entity_id].name_len *32) {
                                    int letter_idx = (int)(offset_to_model_base_x + 32.0f) / 32;
                                    int letter_code = (int)entities[entity_id].name[letter_idx] - 48;
                                    if (letter_code >= 0 && letter_code < 122 - 48) {
                                        unsigned char* letter = (unsigned char*)&device_data_assets[device_data_assets[font_position + (int)entities[entity_id].name[letter_idx] - 48]];
                                        int letter_y = (int)offset_to_model_base_y + 32;
                                        int letter_x = ((int)offset_to_model_base_x + 32) % 32;
                                        float letter_alpha = letter[letter_y * (32 * 4) + letter_x * 4 + 3];
                                        if (letter_alpha > 25) {
                                            //printf("%i ", (int)entities[entity_id].name[letter_idx]);                                        
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - letter_alpha) / 255.0f * 255 + (letter_alpha / 255.0f) * letter[letter_y * (32 * 4) + letter_x * 4 + current_channel]));
                                        }
                                    }
                                }
                            }

                            if (offset_to_model_base_x >= 1 && offset_to_model_base_x < m->model_dimensions[0] * upscale_fac - 1 &&
                                offset_to_model_base_y >= 1 && offset_to_model_base_y < m->model_dimensions[1] * upscale_fac - 1) {

                                /*
                                  output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 0;
                                  output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255 * (e + 1 % 2);
                                  output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255 * (e % 2);
                                  output[current_y * (output_width * output_channels) + current_x * output_channels + 3] = 100;
                                  */

                                unsigned int* p_model_positions = (unsigned int*)&device_data_assets[model_position];
                                unsigned char* p_model = (unsigned char*)&device_data_assets[p_model_positions[(int)(entities[entity_id].orientation / 10) % 36]];

                                if (m->mt == MT_LOOTABLE_ITEM) {
                                    if (offset_to_model_base_x <= 22 * camera_z || offset_to_model_base_y <= 22 * camera_z || offset_to_model_base_x >= m->model_dimensions[0] * upscale_fac - (22 * camera_z) || offset_to_model_base_y >= m->model_dimensions[1] * upscale_fac - (22 * camera_z)) {
                                        float alpha_item = 150;
                                        if (m->id == 50) {
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = (unsigned char)(((255 - alpha_item) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (alpha_item / 255.0f) * 255));
                                        } else if (m->id == 51) {
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = (unsigned char)(((255 - alpha_item) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (alpha_item / 255.0f) * 255));
                                        } else if (m->id == 52) {
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = (unsigned char)(((255 - alpha_item) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (alpha_item / 255.0f) * 255));
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = (unsigned char)(((255 - alpha_item) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (alpha_item / 255.0f) * 146));
                                        }
                                    }
                                }

                                for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
                                    for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
                                        if (offset_to_model_base_x + s_x >= 1 && offset_to_model_base_x + s_x < m->model_dimensions[0] * upscale_fac - 1 &&
                                            offset_to_model_base_y + s_y >= 1 && offset_to_model_base_y + s_y < m->model_dimensions[1] * upscale_fac - 1
                                            ) {
                                            float model_palette_idx_x = offset_to_model_base_x + s_x;
                                            float model_palette_idx_y = offset_to_model_base_y + s_y;

                                            float interpixel_alpha = getInterpixel(p_model, m->model_dimensions[0] * upscale_fac, m->model_dimensions[1] * upscale_fac, 4, model_palette_idx_x, model_palette_idx_y, 3);
                                            if (interpixel_alpha >= 64) {
                                                if (entities[entity_id].position[1] + (m->model_dimensions[1] * m->model_scale) > player_y_max) {
                                                    player_y_max = entities[entity_id].position[1] + (m->model_dimensions[1] * m->model_scale);
                                                    player_id_max = entity_id;
                                                    s_y = sampling_filter_dim;
                                                    s_x = sampling_filter_dim;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (player_id_max >= 0) {
                    unsigned int entity_id = player_id_max;
                    //for (int e = 0; e < entities_count; e++) {
                        //unsigned int entity_id = device_data_players[entities_iddata_position + 1 + e];
                        //if (entity_id < UINT_MAX) {
                    struct model* m = nullptr;
                    if (entities[entity_id].et == ET_PLAYER) {
                        m = &player_models[entities[entity_id].model_id];
                    }
                    else if (entities[entity_id].et == ET_ITEM) {
                        m = &item_models[entities[entity_id].model_id - 50];
                    }
                    else if (entities[entity_id].et == ET_STATIC_ASSET) {
                        m = &map_models[entities[entity_id].model_id - 100];
                    }

                    if (m != nullptr) {
                        const unsigned int* model_positions = &device_data_assets[m->model_positions];
                        int upscale = 1;
                        float upscale_fac = 1.0f;
                        unsigned int model_position = model_positions[upscale - 1];
                        while (camera_z / (m->model_scale / upscale_fac) < 2 && upscale - 1 < m->model_zoom_level_count - 1) {
                            upscale++;
                            model_position = model_positions[upscale - 1];
                            upscale_fac *= 2.0f;
                        }

                        sampling_filter_dim = ceilf(camera_z / (m->model_scale / upscale_fac));

                        float offset_to_model_base_x = (current_game_x - (entities[entity_id].position[0])) / (m->model_scale / upscale_fac);
                        float offset_to_model_base_y = (current_game_y - (entities[entity_id].position[1])) / (m->model_scale / upscale_fac);

                        if (offset_to_model_base_x >= 1 && offset_to_model_base_x < m->model_dimensions[0] * upscale_fac - 1 &&
                            offset_to_model_base_y >= 1 && offset_to_model_base_y < m->model_dimensions[1] * upscale_fac - 1) {

                            unsigned int* p_model_positions = (unsigned int*)&device_data_assets[model_position];
                            unsigned char* p_model = (unsigned char*)&device_data_assets[p_model_positions[(int)(entities[entity_id].orientation / 10) % 36]];

                            for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
                                for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
                                    if (offset_to_model_base_x + s_x >= 1 && offset_to_model_base_x + s_x < m->model_dimensions[0] * upscale_fac - 1 &&
                                        offset_to_model_base_y + s_y >= 1 && offset_to_model_base_y + s_y < m->model_dimensions[1] * upscale_fac - 1
                                        ) {
                                        float model_palette_idx_x = offset_to_model_base_x + s_x;
                                        float model_palette_idx_y = offset_to_model_base_y + s_y;

                                        float interpixel_alpha = getInterpixel(p_model, m->model_dimensions[0] * upscale_fac, m->model_dimensions[1] * upscale_fac, 4, model_palette_idx_x, model_palette_idx_y, 3);
                                        if (interpixel_alpha > 0) {
                                            /*
                                            if (m->mt == MT_PLAYER && interpixel_alpha < 255) {
                                                if (abs(current_mouse_game_x - entities[entity_id].position[0]) <= 32 && abs(current_mouse_game_y - entities[entity_id].position[1]) <= 32) {
                                                    output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (interpixel_alpha / 255.0f) * 200));
                                                }
                                            }
                                            */
                                            float interpixel = getInterpixel(p_model, m->model_dimensions[0] * upscale_fac, m->model_dimensions[1] * upscale_fac, 4, model_palette_idx_x, model_palette_idx_y, current_channel);
                                            output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (interpixel_alpha / 255.0f) * interpixel));
                                        }
                                    }
                                }
                            }
                        }
                        //}
                    }
                }
            }
        }
    }
}

void launch_draw_entities_kernel(
    const unsigned int* device_data_assets, const unsigned int players_models_position, const unsigned int item_models_position, const unsigned int map_models_position, const unsigned int font_position,
    const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int gd_position_in_bf, const unsigned int gd_data_position_in_bf,
    unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const struct vector2<unsigned int> mouse_position, const unsigned int tick_counter) {

    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_width * output_height * 3 + threadsPerBlock - 1) / threadsPerBlock;

    draw_entities_kernel << <blocksPerGrid, threadsPerBlock >> > (device_data_assets, players_models_position, item_models_position, map_models_position, font_position,
                                device_data_rw, entities_position, gd_position_in_bf, gd_data_position_in_bf, 
                                device_data_output, output_position, output_width, output_height, output_channels, 
                                camera_x1, camera_y1, camera_z, mouse_position, tick_counter);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in draw_entities_kernel (error code %s)\n", cudaGetErrorString(err));
    }
}

void entity_add(string name, enum entity_type et, unsigned int model_id, unsigned int model_z) {
    struct entity e;
    e.et = et;
    e.force = { 0.0f, 0.0f };
    e.velocity = { 0.0f, 0.0f };
    for (int i = 0; i < name.length() && i < 50; i++) {
        e.name[i] = name[i];
        e.name_len = i+1;
    }
    for (int i = name.length(); i < 50; i++) {
        e.name[i] = '\0';
    }
    e.orientation = (float)(rand() % 360);
    e.model_id = model_id;
    e.model_z = model_z;
    for (int i = 0; i < 50; i++) {
        e.params[i] = 0;
    }
    entities.push_back(e);
}

void entities_upload(struct bit_field* bf) {
    unsigned int size = entities.size() * sizeof(struct entity);
    entities_size_in_bf = (unsigned int)ceilf(size / (float)sizeof(unsigned int));
    entities_position = bit_field_add_bulk(bf, (unsigned int *) entities.data(), entities_size_in_bf, size)+1;
}