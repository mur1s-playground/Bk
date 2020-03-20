#include "UI.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "AssetLoader.hpp"
#include "KillFeed.hpp"
#include "Util.hpp"
#include <utility>
#include <SDL.h>
#include "Entity.hpp"
#include "Game.hpp"
#include "Grid.hpp"
#include "Main.hpp"
#include "MapEditor.hpp"
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;

string                  ui_active = "";
map<string, struct ui>  uis;

int                         ui_active_id = -1;
map<string, unsigned int>   ui_elements_position;
unsigned int		    	ui_fonts_position = 0;

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

__forceinline__
__device__ void sampling(const unsigned char* frame, const unsigned int width, const unsigned int height, float x, float y, const float alpha_threshold, const float sampling_fac, const unsigned int sampling_filter_dim, unsigned char* target_frame, const unsigned int target_width, const unsigned int target_x, const unsigned int target_y) {
    for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
        for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
            float src_s_x = x * sampling_fac + s_x;
            float src_s_y = y * sampling_fac + s_y;
            if (src_s_x >= 0 && src_s_x < width && src_s_y >= 0 && src_s_y < height) {
                float interpixel_alpha = getInterpixel(frame, width, height, 4, src_s_x, src_s_y, 3);
                if (interpixel_alpha > alpha_threshold) {
                    for (int c = 0; c < 3; c++) {
                        float interpixel = getInterpixel(frame, width, height, 4, src_s_x, src_s_y, c);
                        target_frame[target_y * (target_width * 4) + target_x * 4 + c] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * target_frame[target_y * (target_width * 4) + target_x * 4 + c] + (interpixel_alpha / 255.0f) * interpixel));
                    }
                }
            }
        }
    }
}

__global__ void draw_ui_kernel(const unsigned int* bf_assets_data, const unsigned int background_position, const unsigned int fonts_position, unsigned int* bf_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels, const unsigned int *bf_rw_data, const unsigned int ui_elements_position, const unsigned int tick_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height) {
        int current_x = (i % width);
        int current_y = (i / width);

        unsigned char* target_frame = (unsigned char*)&bf_output_data[output_position];
        unsigned char* frame = (unsigned char*)&bf_assets_data[background_position];

        float alpha = frame[current_y * (width * channels) + current_x * channels + 3];
        if (alpha > 0) {
            for (int c = 0; c < channels - 1; c++) {
                float pixel = frame[current_y * (width * channels) + current_x * channels + c];
                target_frame[current_y * (width * channels) + current_x * channels + c] = (unsigned char)(((255 - alpha) / 255.0f * target_frame[current_y * (width * channels) + current_x * channels + c] + (alpha / 255.0f) * pixel));
            }
        }
        if (ui_elements_position > 0) {
            unsigned int uies_count = (unsigned int)bf_rw_data[ui_elements_position - 1] / ceilf(sizeof(struct ui_element)/(float) sizeof(unsigned int));
            struct ui_element* uies = (struct ui_element*) &bf_rw_data[ui_elements_position];
            for (int ui = 0; ui < uies_count; ui++) {
                struct ui_element *uie = &uies[ui];
                if (uie->uet == UET_TEXTFIELD) {
                    int name_len = 0;
                    for (int j = 0; j < 50; j++) {
                        if (uie->value[j] == '\0') {
                            break;
                        }
                        name_len++;
                    }
                    int fsize = uie->font_size;
                    int fsize_fac = 32 / fsize;
                    if (current_x >= uie->x1y1[0] && current_x < uie->x1y1[0] + fsize * name_len && current_x < uie->x2y2[0] && current_y >= uie->x1y1[1] && current_y < uie->x1y1[1] + fsize) {
                        int letter_idx = (current_x - uie->x1y1[0]) / fsize;
                        int letter_x = (current_x - uie->x1y1[0]) % fsize;
                        int letter_y = (current_y - uie->x1y1[1]) % fsize;
                        if (uie->value[letter_idx] != '\0') {
                            unsigned int letter_frame_pos = bf_assets_data[fonts_position + (int)uie->value[letter_idx]];
                            unsigned char* letter_frame = (unsigned char*)&bf_assets_data[letter_frame_pos];
                            float alpha = letter_frame[(letter_y)*fsize_fac * (32 * 4) + (letter_x * fsize_fac) * 4 + 3];
                            if (alpha > 0) {
                                for (int c = 0; c < channels - 1; c++) {
                                    target_frame[current_y * (width * channels) + current_x * channels + c] = (unsigned char)((255 - alpha) / 255.0f * target_frame[current_y * (width * channels) + current_x * channels + c] + (alpha / 255.0f) * letter_frame[(letter_y)*fsize_fac * (32 * 4) + letter_x * fsize_fac * 4 + c]);
                                }
                            }
                        }
                    }
                    /*
                    float fsize_fac = 32 / (float)fsize;
                    if (current_x >= uie->x1y1[0] && current_x < uie->x1y1[0]+fsize*name_len && current_x < uie->x2y2[0] && current_y >= uie->x1y1[1] && current_y < uie->x1y1[1]+fsize) {
                        int letter_idx = (current_x - uie->x1y1[0]) / fsize;
                        float letter_x = current_x - uie->x1y1[0] - (letter_idx*fsize);
                        float letter_y = (current_y - uie->x1y1[1]) * fsize_fac;
                        if (uie->value[letter_idx] != '\0') {

                            int sampling_filter_dim = (int)ceilf(fsize_fac);

                            unsigned int letter_frame_pos = bf_assets_data[fonts_position + (int)uie->value[letter_idx] - 48];
                            unsigned char* letter_frame = (unsigned char*)&bf_assets_data[letter_frame_pos];

                            for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
                                for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
                                    float letter_s_x = letter_x*fsize_fac + s_x;
                                    float letter_s_y = letter_y*fsize_fac + s_y;
                                    if (letter_s_x >= 0 && letter_s_x < 32 && letter_s_y >= 0 && letter_s_y < 32) {
                                        float interpixel_alpha = getInterpixel(letter_frame, 32, 32, 4, letter_s_x, letter_s_y, 3);
                                        if (interpixel_alpha > 25) {
                                            for (int c = 0; c < channels - 1; c++) {
                                                float interpixel = getInterpixel(letter_frame, 32, 32, 4, letter_s_x, letter_s_y, c);
                                                target_frame[current_y * (width * channels) + current_x * channels + c] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * target_frame[current_y * (width * channels) + current_x * channels + c] + (interpixel_alpha / 255.0f) * interpixel));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    */
                } else if (uie->uet == UET_SCROLLLIST) {
                    if (current_x >= uie->x1y1[0] && current_x < uie->x2y2[0] && current_y >= uie->x1y1[1] && current_y < uie->x2y2[1]) {
                        int* config = (int*)&uie->value[0];

                        int content_position        = config[0];
                        int content_lines           = config[1];
                        int content_line_length     = config[2];
                        int content_order           = config[3];
                        int content_align           = config[4];
                        int content_scroll_pos      = config[5];
                        //sgroup                    = config[6]
                        int content_fsize           = config[7];
                        unsigned char* content_fcol = (unsigned char *)&config[8];

                        unsigned char* kfes = (unsigned char *)&bf_rw_data[content_position];
                        //int line_count = (bf_rw_data[content_position - 1] * sizeof(unsigned int)) / content_line_length;

                        if (content_scroll_pos + (current_y - uie->x1y1[1]) / (content_fsize+2) < content_lines) {
                            int current_line_y = content_scroll_pos + ((current_y - uie->x1y1[1]) / (content_fsize+2));
                            if (content_order == 0) {
                               current_line_y = content_lines - 1 - current_line_y;
                            }

                            unsigned char* current_kfe = &kfes[current_line_y * content_line_length];
                            int text_len = 0;
                            for (int j = 0; j < content_line_length; j++) {
                                if (current_kfe[j] == '\0') break;
                                text_len++;
                            }

                            int letter_idx_max = (uie->x2y2[0] - uie->x1y1[0]) / content_fsize;

                            int scroll_width = text_len - letter_idx_max;
                            float scroll_shift;
                            if (scroll_width > 0) {
                                scroll_shift = fmod(tick_counter/30.0f, (float)scroll_width);
                            } else {
                                scroll_shift = 0;
                            }

                            int letter_idx = current_x;
                            int letter_x = current_x;
                            int letter_x_delta = 0;
                            if (text_len > letter_idx_max || content_align == 0) {
                                letter_x_delta = ((int)floorf((scroll_shift - (int)floorf(scroll_shift)) * (float)content_fsize));
                                letter_idx += (int)floorf(scroll_shift)*content_fsize + letter_x_delta;
                                letter_x += letter_x_delta;
                                letter_idx = (letter_idx - uie->x1y1[0]) / content_fsize;
                                letter_x = (letter_x - uie->x1y1[0]) % content_fsize;
                            } else {
                                letter_idx = letter_idx_max - (letter_idx - uie->x1y1[0]) / content_fsize;
                                letter_x = (letter_x - uie->x1y1[0]) % content_fsize;
                            }

                            int letter_y = (current_y - uie->x1y1[1]) % (content_fsize+2);

                            int fontsize_fac = 32 / content_fsize;
                            if (letter_y > 0 && letter_y < content_fsize+1) {
                                if (letter_idx < text_len && letter_idx >= 0) {
                                    if (current_kfe[letter_idx] != '\0') {
                                        unsigned int letter_frame_pos;
                                        if (text_len > letter_idx_max || content_align == 0) {
                                            letter_frame_pos = bf_assets_data[fonts_position + (int)current_kfe[letter_idx]];
                                        } else {
                                            letter_frame_pos = bf_assets_data[fonts_position + (int)current_kfe[text_len - 1 - letter_idx]];
                                        }
                                        unsigned char* letter_frame = (unsigned char*)&bf_assets_data[letter_frame_pos];

                                        //int sampling_filter_dim = (int)ceilf(fontsize_fac);
                                        //sampling(letter_frame, 32, 32, letter_x, letter_y - 1, 25, fontsize_fac, sampling_filter_dim, target_frame, width, current_x, current_y);
                                            
                                        float alpha = letter_frame[((letter_y - 1) * fontsize_fac) * (32 * 4) + (letter_x * fontsize_fac) * 4 + 3];
                                        if (alpha > 0) {
                                            for (int c = 0; c < channels - 1; c++) {
                                                unsigned char col;
                                                if (content_fcol[c] == 0) {
                                                    col = letter_frame[((letter_y - 1) * fontsize_fac) * (32 * 4) + (letter_x * fontsize_fac) * 4 + c];
                                                } else {
                                                    col = content_fcol[c];
                                                }
                                                target_frame[current_y * (width * channels) + current_x * channels + c] = (unsigned char)((255 - alpha) / 255.0f * target_frame[current_y * (width * channels) + current_x * channels + c] + (alpha / 255.0f) * col);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }          
                } else if (uie->uet == UET_IMAGE) {
                    if (current_x >= uie->x1y1[0] && current_x < uie->x2y2[0] && current_y >= uie->x1y1[1] && current_y < uie->x2y2[1]) {
                        int* config = (int*)&uie->value[0];

                        int image_positions = config[0];
                        int animation_ticks = config[3];
                        int animation_stepsize = config[4];

                        int image_position = image_positions;
                        if (animation_ticks > 1) {
                            int image_idx = (tick_counter / animation_stepsize) % animation_ticks;
                            image_position = bf_assets_data[config[0] + image_idx];
                        }

                        if (image_position > 0) {
                            int image_width = config[1];
                            int image_height = config[2];

                            int space_width = uie->x2y2[0] - uie->x1y1[0];
                            int space_height = uie->x2y2[1] - uie->x1y1[1];
                            float image_scale = 1.0f;
                            if (image_width > space_width) {
                                image_scale = space_width / (float)image_width;
                            }
                            if (image_height > space_height) {
                                if (space_height / (float)image_height < image_scale) {
                                    image_scale = space_height / (float)image_height;
                                }
                            }
                            unsigned char* src_frame = (unsigned char*)&bf_assets_data[image_position];

                            float image_source_x = current_x - uie->x1y1[0];
                            float image_source_y = current_y - uie->x1y1[1];

                            int sampling_filter_dim = (int)ceilf(1.0f/image_scale);

                            sampling(src_frame, image_width, image_height, image_source_x, image_source_y, 25, 1.0f/image_scale, sampling_filter_dim, target_frame, width, current_x, current_y);
                        }
                    }
                }
            }
        }
    }
}

void launch_draw_ui_kernel(const unsigned int *bf_assets_data, const unsigned int background_position, const unsigned int font_position, unsigned int *bf_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels, const unsigned int* bf_rw_data, const unsigned int tick_counter) {
    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int ui_element_position = 0;
    if (ui_active != "") {
        ui_element_position = ui_elements_position[ui_active];
    }

    draw_ui_kernel<<<blocksPerGrid, threadsPerBlock>>>(bf_assets_data, background_position, font_position, bf_output_data, output_position, width, height, channels, bf_rw_data, ui_element_position, tick_counter);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in draw_ui_kernel (error code %s)\n", cudaGetErrorString(err));
    }
}

void ui_init(struct bit_field* bf_assets, struct bit_field *bf_rw) {
    asset_loader_load_folder(bf_assets, "./font/");
    vector<unsigned int> font_pos;
    for (int i = 0; i <= 127; i++) {
        stringstream ch;
        ch << i;
        font_pos.emplace_back(assets["./font/" + ch.str() + ".png"]);
    }
    ui_fonts_position = bit_field_add_bulk(bf_assets, font_pos.data(), font_pos.size(), font_pos.size()*sizeof(unsigned int))+1;
    vector<string> ui_cfgs = get_all_files_names_within_folder("./ui", "*", "cfg");
    for (int c = 0; c < ui_cfgs.size(); c++) {
        size_t pos = ui_cfgs[c].find_last_of(".");
        if (pos != string::npos) {
            string content_folder = ui_cfgs[c].substr(0, pos);
            vector<pair<string, string>> kv_pairs = get_cfg_key_value_pairs("./ui/", ui_cfgs[c]);
            asset_loader_load_folder(bf_assets, "./ui/" + content_folder + "/");
            struct ui u;
            u.active_element_id = -1;
            for (int i = 0; i < kv_pairs.size(); i++) {
                if (kv_pairs[i].first == "background") {
                    u.background_position = assets["./ui/" + content_folder + "/" + kv_pairs[i].second + ".png"];
                }
                if (kv_pairs[i].first == "textfield") {
                    struct ui_element t;
                    t.uet = UET_TEXTFIELD;
                    t.name = kv_pairs[i].second;
                    for (int ch = 0; ch < 51; ch++) {
                        t.value[ch] = '\0';
                    }
                    if (kv_pairs[i + 1].first == kv_pairs[i].second + "_x1y1") {
                        size_t sep = kv_pairs[i + 1].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 1].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 1].second.substr(sep + 1);
                            second = trim(second);
                            t.x1y1 = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                        }
                    }
                    if (kv_pairs[i + 2].first == kv_pairs[i].second + "_x2y2") {
                        size_t sep = kv_pairs[i + 2].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 2].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 2].second.substr(sep + 1);
                            second = trim(second);
                            t.x2y2 = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                        }
                    }
                    if (kv_pairs[i + 3].first == kv_pairs[i].second + "_fsize") {
                        string first = kv_pairs[i + 3].second;
                        first = trim(first);
                        t.font_size = stoi(first);
                    } else {
                        t.font_size = 32;
                    }
                    printf("adding textfield %s\n", t.name.c_str());
                    u.ui_elements.push_back(t);
                }
                if (kv_pairs[i].first == "button") {
                    struct ui_element b;
                    b.uet = UET_BUTTON;
                    b.name = kv_pairs[i].second;
                    if (kv_pairs[i + 1].first == kv_pairs[i].second + "_x1y1") {
                        size_t sep = kv_pairs[i+1].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 1].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 1].second.substr(sep + 1);
                            second = trim(second);
                            b.x1y1 = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                        }
                    }
                    if (kv_pairs[i + 2].first == kv_pairs[i].second + "_x2y2") {
                        size_t sep = kv_pairs[i + 2].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 2].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 2].second.substr(sep + 1);
                            second = trim(second);
                            b.x2y2 = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                        }
                    }
                    if (kv_pairs[i + 3].first == kv_pairs[i].second + "_action") {
                        size_t sep = kv_pairs[i + 3].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 3].second.substr(0, sep);
                            first = trim(first);
                            if (first == "UI") {
                                b.ocat = BAT_UI;
                            } else if (first == "GS") {
                                b.ocat = BAT_GS;
                            }
                            string second = kv_pairs[i + 3].second.substr(sep + 1);
                            second = trim(second);
                            b.ocap = second;
                        } else {
                            string first = kv_pairs[i + 3].second;
                            first = trim(first);
                            if (first == "QUIT") {
                                b.ocat = BAT_QUIT;
                            } else if (first == "GAMESTART") {
                                b.ocat = BAT_GAMESTART;
                            } else if (first == "GAMEEND") {
                                b.ocat = BAT_GAMEEND;
                            } else if (first == "MAPEDITOR_SAVEQUIT") {
                                b.ocat = BAT_MAPEDITOR_SAVEQUIT;
                            }
                        }
                    }
                    printf("adding button %s\n", b.name.c_str());
                    u.ui_elements.push_back(b);
                }
                if (kv_pairs[i].first == "scrolllist") {
                    struct ui_element t;
                    t.uet = UET_SCROLLLIST;
                    t.name = kv_pairs[i].second;
                    for (int ch = 0; ch < 51; ch++) {
                        t.value[ch] = '\0';
                    }
                    int* config = (int*)&t.value[0];
                    if (kv_pairs[i + 1].first == kv_pairs[i].second + "_x1y1") {
                        size_t sep = kv_pairs[i + 1].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 1].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 1].second.substr(sep + 1);
                            second = trim(second);
                            t.x1y1 = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                        }
                    }
                    if (kv_pairs[i + 2].first == kv_pairs[i].second + "_x2y2") {
                        size_t sep = kv_pairs[i + 2].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 2].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 2].second.substr(sep + 1);
                            second = trim(second);
                            t.x2y2 = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                        }
                    }
                    config[1] = 0;
                    if (kv_pairs[i + 3].first == kv_pairs[i].second + "_chars") {
                        string first = kv_pairs[i + 3].second;
                        first = trim(first);
                        config[2] = stoi(first);
                    }
                    if (kv_pairs[i + 4].first == kv_pairs[i].second + "_asc") {
                        string first = kv_pairs[i + 4].second;
                        first = trim(first);
                        config[3] = stoi(first);
                    }
                    if (kv_pairs[i + 5].first == kv_pairs[i].second + "_align") {
                        string first = kv_pairs[i + 5].second;
                        first = trim(first);
                        config[4] = stoi(first);
                    }
                    if (kv_pairs[i + 6].first == kv_pairs[i].second + "_sgroup") {
                        string first = kv_pairs[i + 6].second;
                        first = trim(first);
                        config[6] = stoi(first);
                    }
                    if (kv_pairs[i + 7].first == kv_pairs[i].second + "_fsize") {
                        string first = kv_pairs[i + 7].second;
                        first = trim(first);
                        config[7] = stoi(first);
                    }
                    if (kv_pairs[i + 8].first == kv_pairs[i].second + "_fcol") {
                        size_t sep = kv_pairs[i + 8].second.find_first_of(",");
                        size_t sep_2 = kv_pairs[i + 8].second.find_last_of(",");
                        if (sep != string::npos && sep_2 != string::npos) {
                            string first = kv_pairs[i + 8].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 8].second.substr(sep + 1, sep_2-sep-1);
                            second = trim(second);
                            string third = kv_pairs[i + 8].second.substr(sep_2 + 1);
                            third = trim(third);
                            unsigned char* col = (unsigned char *)&config[8];
                            col[0] = stoi(first);
                            col[1] = stoi(second);
                            col[2] = stoi(third);
                        }
                    }
                    config[5] = 0;
                    printf("adding scrolllist %s %i %i\n", t.name.c_str(), config[1], config[2]);
                    u.ui_elements.push_back(t);
                }
                if (kv_pairs[i].first == "image") {
                    struct ui_element t;
                    t.uet = UET_IMAGE;
                    t.name = kv_pairs[i].second;
                    for (int ch = 0; ch < 51; ch++) {
                        t.value[ch] = '\0';
                    }
                    int* config = (int*)&t.value[0];
                    if (kv_pairs[i + 1].first == kv_pairs[i].second + "_x1y1") {
                        size_t sep = kv_pairs[i + 1].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 1].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 1].second.substr(sep + 1);
                            second = trim(second);
                            t.x1y1 = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                        }
                    }
                    if (kv_pairs[i + 2].first == kv_pairs[i].second + "_x2y2") {
                        size_t sep = kv_pairs[i + 2].second.find(",");
                        if (sep != string::npos) {
                            string first = kv_pairs[i + 2].second.substr(0, sep);
                            first = trim(first);
                            string second = kv_pairs[i + 2].second.substr(sep + 1);
                            second = trim(second);
                            t.x2y2 = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                        }
                    }
                    config[3] = 1;
                    if (kv_pairs[i + 3].first == kv_pairs[i].second + "_animation_ticks") {
                        string first = kv_pairs[i + 3].second;
                        first = trim(first);
                        config[3] = stoi(first);
                    }
                    config[4] = 1;
                    if (kv_pairs[i + 4].first == kv_pairs[i].second + "_animation_stepsize") {
                        string first = kv_pairs[i + 4].second;
                        first = trim(first);
                        config[4] = stoi(first);
                    }
                    config[0] = 0;
                    config[1] = 0;
                    config[2] = 0;
                    if (kv_pairs[i + 5].first == kv_pairs[i].second + "_file_prefix") {
                        string first = kv_pairs[i + 5].second;
                        first = trim(first);
                        vector<unsigned int> asset_positions;
                        for (int ap = 0; ap < config[3]; ap++) {
                            stringstream ss_ap;
                            ss_ap << ap;
                            asset_positions.push_back(assets["./ui/" + content_folder + "/" + first + ss_ap.str() + ".png"]);
                            if (ap == 0) {
                                config[1] = assets_dimensions["./ui/" + content_folder + "/" + first + ss_ap.str() + ".png"].width;
                                config[2] = assets_dimensions["./ui/" + content_folder + "/" + first + ss_ap.str() + ".png"].height;
                            }
                        }
                        config[0] = bit_field_add_bulk(bf_assets, asset_positions.data(), asset_positions.size(), asset_positions.size()*sizeof(unsigned int))+1;
                    }
                    
                    printf("adding image %s\n", t.name.c_str());
                    u.ui_elements.push_back(t);
                }
            }
            uis.emplace(ui_cfgs[c].substr(0, pos), u);
            if (u.ui_elements.size() > 0) {
                int size = u.ui_elements.size() * sizeof(struct ui_element);
                unsigned int size_in_bf = (unsigned int)ceilf(size/(float)sizeof(unsigned int));
                ui_elements_position.emplace(ui_cfgs[c].substr(0, pos), bit_field_add_bulk(bf_rw, (unsigned int*)u.ui_elements.data(), size_in_bf, size)+1);
            } else {
                ui_elements_position.emplace(ui_cfgs[c].substr(0, pos), 0);
            }
        }
    }
}

void ui_set_active(string name) {
    ui_active = name;
    uis[ui_active].active_element_id = -1;
}

bool ui_process_click(struct bit_field *bf_rw, unsigned int x, unsigned int y) {
    vector<struct ui_element> active_elements = uis[ui_active].ui_elements;
    for (int i = 0; i < active_elements.size(); i++) {
        if (x >= active_elements[i].x1y1[0] && x <= active_elements[i].x2y2[0] &&
            y >= active_elements[i].x1y1[1] && y <= active_elements[i].x2y2[1]) {
            printf("set element active %i\n", i);
            uis[ui_active].active_element_id = i;
            if (active_elements[i].uet == UET_BUTTON) {
                enum on_click_action_type ocat = active_elements[i].ocat;
                string on_click_action_param = active_elements[i].ocap;
                if (ocat == BAT_QUIT) {
                    running = false;
                    break;
                } else if (ocat == BAT_GAMESTART) {
                    ui_set_active("loading_game");
                    break;
                } else if (ocat == BAT_GAMEEND) {
                    ui_set_active("unloading_game");
                    break;
                } else if (ocat == BAT_UI) {
                    ui_set_active(on_click_action_param);
                    break;
                } else if (ocat == BAT_GS) {
                    game_ticks_target = stoi(on_click_action_param);
                    break;
                } else if (ocat == BAT_MAPEDITOR_SAVEQUIT) {
                    mapeditor_save();
                    running = false;
                    break;
                }
            } else if (active_elements[i].uet == UET_TEXTFIELD) {

            } else if (active_elements[i].uet == UET_SCROLLLIST) {
                struct ui_element* uies = (struct ui_element*) &bf_rw->data[ui_elements_position[ui_active]];
                int* config = (int*)&uies[i].value[0];
                int content_lines = config[1];
                int content_order = config[3];
                int content_scroll_pos = config[5];
                int content_fsize = config[7];

                if (content_scroll_pos + ((y - active_elements[i].x1y1[1]) / (content_fsize + 2)) < content_lines) {
                    int current_line_y = content_scroll_pos + ((y - active_elements[i].x1y1[1]) / (content_fsize + 2));
                    if (content_order == 0) {
                        current_line_y = content_lines - 1 - current_line_y;
                    }
                    uis[ui_active].active_element_param = current_line_y;
                } else {
                    uis[ui_active].active_element_param = -1;
                }
            }
            return true;
        }
    }
    return false;
}

bool ui_process_scroll(struct bit_field *bf_rw, unsigned int x, unsigned int y, int z) {
    vector<struct ui_element> active_elements = uis[ui_active].ui_elements;
    int sgroup = 0;
    int fid = -1;
    for (int i = 0; i < active_elements.size(); i++) {
        if (x >= active_elements[i].x1y1[0] && x <= active_elements[i].x2y2[0] &&
            y >= active_elements[i].x1y1[1] && y <= active_elements[i].x2y2[1]) {
            
            if (active_elements[i].uet == UET_SCROLLLIST) {
                fid = i;
                int* config = (int*)&active_elements[i].value[0];
                sgroup = config[6];
                if (z < 0) {
                    if (config[5] < config[1] - 1) {
                        config[5]++;
                        ui_value_as_config(bf_rw, ui_active, active_elements[i].name, 5, config[5]);
                    }
                } else {
                    if (config[5] > 0) {
                        config[5]--;
                        ui_value_as_config(bf_rw, ui_active, active_elements[i].name, 5, config[5]);
                    }
                }
                if (sgroup == 0) return true;
            }
        }
    }
    if (sgroup > 0) {
        for (int i = 0; i < active_elements.size(); i++) {
            if (i != fid) {
                if (active_elements[i].uet == UET_SCROLLLIST) {
                    int* config = (int*)&active_elements[i].value[0];
                    int cur_sgroup = config[6];
                    if (cur_sgroup == sgroup) {
                        if (z < 0) {
                            if (config[5] < config[1] - 1) {
                                config[5]++;
                                ui_value_as_config(bf_rw, ui_active, active_elements[i].name, 5, config[5]);
                            }
                        } else {
                            if (config[5] > 0) {
                                config[5]--;
                                ui_value_as_config(bf_rw, ui_active, active_elements[i].name, 5, config[5]);
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
    return false;
}

void ui_process_keys(struct bit_field* bf_rw, const unsigned int x, const unsigned int y, const unsigned int sdl_keyval_enum) {
    if (ui_active != "" && uis[ui_active].active_element_id >= 0) {
        if (uis[ui_active].ui_elements[uis[ui_active].active_element_id].uet == UET_TEXTFIELD) {
            //printf("textfield active with value %s\n", val.c_str());
            if (sdl_keyval_enum == SDLK_BACKSPACE) {
                if (uis[ui_active].ui_elements[uis[ui_active].active_element_id].value[0] != '\0') {
                    for (int i = 1; i < 51; i++) {
                        if (uis[ui_active].ui_elements[uis[ui_active].active_element_id].value[i] == '\0') {
                            uis[ui_active].ui_elements[uis[ui_active].active_element_id].value[i - 1] = '\0';
                            break;
                        }
                    }
                }
            } else {
                string val = "";
                switch (sdl_keyval_enum) {
                    case SDLK_0: val += "0"; break;
                    case SDLK_1: val += "1"; break;
                    case SDLK_2: val += "2"; break;
                    case SDLK_3: val += "3"; break;
                    case SDLK_4: val += "4"; break;
                    case SDLK_5: val += "5"; break;
                    case SDLK_6: val += "6"; break;
                    case SDLK_7: val += "7"; break;
                    case SDLK_8: val += "8"; break;
                    case SDLK_9: val += "9"; break;
                    case SDLK_a: val += "a"; break;
                    case SDLK_b: val += "b"; break;
                    case SDLK_c: val += "c"; break;
                    case SDLK_d: val += "d"; break;
                    case SDLK_e: val += "e"; break;
                    case SDLK_f: val += "f"; break;
                    case SDLK_g: val += "g"; break;
                    case SDLK_h: val += "h"; break;
                    case SDLK_i: val += "i"; break;
                    case SDLK_j: val += "j"; break;
                    case SDLK_k: val += "k"; break;
                    case SDLK_l: val += "l"; break;
                    case SDLK_m: val += "m"; break;
                    case SDLK_n: val += "n"; break;
                    case SDLK_o: val += "o"; break;
                    case SDLK_p: val += "p"; break;
                    case SDLK_q: val += "q"; break;
                    case SDLK_r: val += "r"; break;
                    case SDLK_s: val += "s"; break;
                    case SDLK_t: val += "t"; break;
                    case SDLK_u: val += "u"; break;
                    case SDLK_v: val += "v"; break;
                    case SDLK_w: val += "w"; break;
                    case SDLK_x: val += "x"; break;
                    case SDLK_y: val += "y"; break;
                    case SDLK_z: val += "z"; break;
                    case SDLK_PERIOD: val += "."; break;
                    //TODO: FIXME
                    case SDLK_MINUS: val += "_"; break;
                    case SDLK_UNDERSCORE: val += "_"; break;
                    default: break;
                }
                if (val.length() > 0) {
                    for (int i = 0; i < 50; i++) {
                        if (uis[ui_active].ui_elements[uis[ui_active].active_element_id].value[i] == '\0') {
                            uis[ui_active].ui_elements[uis[ui_active].active_element_id].value[i] = val.c_str()[0];
                            break;
                        }
                    }
                }
            }
            struct ui_element* uies = (struct ui_element*) & bf_rw->data[ui_elements_position[ui_active]];
            struct ui_element* uie = &uies[uis[ui_active].active_element_id];
            for (int v = 0; v < 50; v++) {
                uie->value[v] = uis[ui_active].ui_elements[uis[ui_active].active_element_id].value[v];
            }
            int size = sizeof(struct ui_element);
            unsigned int size_in_bf = (unsigned int)ceilf(size / (float)sizeof(unsigned int));
            bit_field_invalidate_bulk(bf_rw, ui_elements_position[ui_active] + uis[ui_active].active_element_id * size_in_bf, size_in_bf);
            return;
            //printf("new value of textfield is %s\n", uis[ui_active].ui_elements[uis[ui_active].active_element_id].value);
        }
    }
    if (ui_active != "") {
        vector<struct ui_element> active_elements = uis[ui_active].ui_elements;

        int sgroup = 0;
        int fid = -1;

        for (int i = 0; i < active_elements.size(); i++) {
            if (x >= active_elements[i].x1y1[0] && x <= active_elements[i].x2y2[0] &&
                y >= active_elements[i].x1y1[1] && y <= active_elements[i].x2y2[1]) {
                if (active_elements[i].uet == UET_SCROLLLIST) {
                    fid = i;
                    int* config = (int*)&active_elements[i].value[0];
                    sgroup = config[6];
                    if (sdl_keyval_enum == SDLK_PAGEDOWN){
                        config[5] += 10;
                        if (config[5] >= config[1] - 1) {
                            if (config[1] > 0) {
                                config[5] = config[1] - 1;
                            } else {
                                config[5] = 0;
                            }
                        }
                        ui_value_as_config(bf_rw, ui_active, active_elements[i].name, 5, config[5]);
                    } else if (sdl_keyval_enum == SDLK_PAGEUP) {
                        config[5] -= 10;
                        if (config[5] < 0) config[5] = 0;
                        ui_value_as_config(bf_rw, ui_active, active_elements[i].name, 5, config[5]);
                    }
                    if (sgroup == 0) return;
                }
            }
        }
        if (sgroup > 0) {
            for (int i = 0; i < active_elements.size(); i++) {
                if (i != fid) {
                    if (active_elements[i].uet == UET_SCROLLLIST) {
                        int* config = (int*)&active_elements[i].value[0];
                        int cur_sgroup = config[6];
                        if (cur_sgroup == sgroup) {
                            if (sdl_keyval_enum == SDLK_PAGEDOWN) {
                                config[5] += 10;
                                if (config[5] >= config[1] - 1) {
                                    if (config[1] > 0) {
                                        config[5] = config[1] - 1;
                                    } else {
                                        config[5] = 0;
                                    }
                                }
                                ui_value_as_config(bf_rw, ui_active, active_elements[i].name, 5, config[5]);
                            }
                            else if (sdl_keyval_enum == SDLK_PAGEUP) {
                                config[5] -= 10;
                                if (config[5] < 0) config[5] = 0;
                                ui_value_as_config(bf_rw, ui_active, active_elements[i].name, 5, config[5]);
                            }
                        }
                    }
                }
            }
        }
    }
}

void ui_value_as_config(struct bit_field *bf_rw, string ui_name, string element_name, int index, int value) {
    int uc;
    for (uc = 0; uc < uis[ui_name].ui_elements.size(); uc++) {
        if (uis[ui_name].ui_elements[uc].name == element_name) {
            break;
        }
    }

    if (uc < uis[ui_name].ui_elements.size()) {
        struct ui_element* uies = (struct ui_element*) & bf_rw->data[ui_elements_position[ui_name]];
        struct ui_element* uie = &uies[uc];

        int* config = (int*)&uis[ui_name].ui_elements[uc].value[0];
        config[index] = value;
        int* pos_value = (int*)&uie->value[0];
        pos_value[index] = value;

        int ui_size = uis[ui_name].ui_elements.size() * sizeof(struct ui_element);
        unsigned int ui_size_in_bf = (unsigned int)ceilf(ui_size / (float)sizeof(unsigned int));
        bit_field_invalidate_bulk(bf_rw, ui_elements_position[ui_name], ui_size_in_bf);
    } else {
        printf("trying to write cfg out of ui_elements bounds\n");
    }
}

void ui_textfield_set_int(struct bit_field *bf_rw, string ui_name, string ui_element_name, int value) {
    struct ui_element* uies = (struct ui_element*) &bf_rw->data[ui_elements_position[ui_name]];
    int uc;
    for (uc = 0; uc < uis[ui_name].ui_elements.size(); uc++) {
        if (uis[ui_name].ui_elements[uc].name == ui_element_name) {
            struct ui_element* uie = &uies[uc];
            stringstream ss_val;
            ss_val << value;
            int ca;
            for (ca = 0; ca < ss_val.str().length(); ca++) {
                uie->value[ca] = ss_val.str().at(ca);
                uis[ui_name].ui_elements[uc].value[ca] = ss_val.str().at(ca);
            }
            for (; ca < 50; ca++) {
                if (uie->value[ca] == '\0') break;
                uis[ui_name].ui_elements[uc].value[ca] = '\0';
                uie->value[ca] = '\0';
            }
            break;
        }
    }
    int size = sizeof(struct ui_element);
    unsigned int size_in_bf = (unsigned int)ceilf(size / (float)sizeof(unsigned int));
    bit_field_invalidate_bulk(bf_rw, ui_elements_position[ui_name] + uc * size_in_bf, size_in_bf);
}

string ui_textfield_get_value(struct bit_field* bf_rw, string ui_name, string ui_element_name) {
    struct ui_element* uies = (struct ui_element*) & bf_rw->data[ui_elements_position[ui_name]];
    int uc;
    string val = "";
    for (uc = 0; uc < uis[ui_name].ui_elements.size(); uc++) {
        if (uis[ui_name].ui_elements[uc].name == ui_element_name) {
            for (int i = 0; i < 52; i++) {
                if (uis[ui_name].ui_elements[uc].value[i] == '\0') break;
                val += uis[ui_name].ui_elements[uc].value[i];
            }
            break;
        }
    }
    return val;
}

void ui_textfield_set_value(struct bit_field* bf_rw, string ui_name, string ui_element_name, const char value[50]) {
    struct ui_element* uies = (struct ui_element*) & bf_rw->data[ui_elements_position[ui_name]];
    int uc;
    for (uc = 0; uc < uis[ui_name].ui_elements.size(); uc++) {
        if (uis[ui_name].ui_elements[uc].name == ui_element_name) {
            struct ui_element* uie = &uies[uc];
            int ca;
            for (ca = 0; ca < 50; ca++) {
                if (value[ca] == '\0') break;
                uis[ui_name].ui_elements[uc].value[ca] = value[ca];
                uie->value[ca] = value[ca];
            }
            for (; ca < 50; ca++) {
                if (uie->value[ca] == '\0') break;
                uis[ui_name].ui_elements[uc].value[ca] = '\0';
                uie->value[ca] = '\0';
            }
            break;
        }
    }
    int size = sizeof(struct ui_element);
    unsigned int size_in_bf = (unsigned int)ceilf(size / (float)sizeof(unsigned int));
    bit_field_invalidate_bulk(bf_rw, ui_elements_position[ui_name] + uc * size_in_bf, size_in_bf);
}

void ui_save_fields_to_file(struct bit_field *bf_rw, string ui_name, string ui_fields[], unsigned int field_count, string folder, string filename) {
    string filepath = folder + "/" + filename;

    struct ui_element* uies = (struct ui_element*) & bf_rw->data[ui_elements_position[ui_name]];

    ofstream sfile;
    sfile.open(filepath);
    for (int i = 0; i < field_count; i++) {
        for (int uc = 0; uc < uis[ui_name].ui_elements.size(); uc++) {
            if (uis[ui_name].ui_elements[uc].name == ui_fields[i]) {
                sfile << "ui\t:\t" << ui_name << std::endl;
                sfile << ui_name << "_field\t:\t" << ui_fields[i] << std::endl;
                sfile << ui_name << "_value\t:\t";
                for (int ch = 0; ch < 50; ch++) {
                    if (uis[ui_name].ui_elements[uc].value[ch] == '\0') break;
                    sfile << uis[ui_name].ui_elements[uc].value[ch];
                }
                sfile << std::endl;
                sfile << std::endl;
            }
        }
    }
    sfile.close();
}