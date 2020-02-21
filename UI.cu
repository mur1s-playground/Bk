#include "UI.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "AssetLoader.hpp"
#include "Util.hpp"
#include <utility>

using namespace std;

string                  ui_active = "";
map<string, struct ui>  uis;

__global__ void draw_ui_kernel(const unsigned int* bf_assets_data, const unsigned int background_position, unsigned int* bf_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels) {
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
    }
}

void launch_draw_ui_kernel(const unsigned int *bf_assets_data, const unsigned int background_position, unsigned int *bf_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels) {
    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    draw_ui_kernel<<<blocksPerGrid, threadsPerBlock>>>(bf_assets_data, background_position, bf_output_data, output_position, width, height, channels);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in draw_ui_kernel (error code %s)\n", cudaGetErrorString(err));
    }
}

void ui_init(struct bit_field* bf_assets) {
    vector<string> ui_cfgs = get_all_files_names_within_folder("./ui", "*", "cfg");
    for (int c = 0; c < ui_cfgs.size(); c++) {
        size_t pos = ui_cfgs[c].find_last_of(".");
        if (pos != string::npos) {
            string content_folder = ui_cfgs[c].substr(0, pos);
            vector<pair<string, string>> kv_pairs = get_cfg_key_value_pairs("./ui/", ui_cfgs[c]);
            asset_loader_load_folder(bf_assets, "./ui/" + content_folder + "/");
            struct ui u;
            for (int i = 0; i < kv_pairs.size(); i++) {
                if (kv_pairs[i].first == "background") {
                    u.background_position = assets["./ui/" + content_folder + "/" + kv_pairs[i].second + ".png"];
                }
                if (kv_pairs[i].first == "button") {
                    struct button b;
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
                                b.bat = BAT_UI;
                            }
                            string second = kv_pairs[i + 3].second.substr(sep + 1);
                            second = trim(second);
                            b.bap = second;
                        } else {
                            string first = kv_pairs[i + 3].second;
                            first = trim(first);
                            if (first == "QUIT") {
                                b.bat = BAT_QUIT;
                            } else if (first == "GAMESTART") {
                                b.bat = BAT_GAMESTART;
                            }
                        }
                    }
                    u.buttons.push_back(b);
                    i++;
                    i++;
                    i++;
                }
            }
            uis.emplace(ui_cfgs[c].substr(0, pos), u);
        }
    }
}

void ui_set_active(string name) {
    ui_active = name;
}

void ui_process_click(unsigned int x, unsigned int y) {
    vector<struct button> active_buttons = uis[ui_active].buttons;
    for (int i = 0; i < active_buttons.size(); i++) {
        if (x >= active_buttons[i].x1y1[0] && x <= active_buttons[i].x2y2[0] &&
            y >= active_buttons[i].x1y1[1] && y <= active_buttons[i].x2y2[1]) {
            enum button_action_type current_action = active_buttons[i].bat;
            string current_action_param = active_buttons[i].bap;
            if (current_action == BAT_QUIT) {
                exit(0);
            } else if (current_action == BAT_GAMESTART) {
                //TODO: process
                break;
            } else if (current_action == BAT_UI) {
                ui_set_active(current_action_param);
                break;
            }
        }
    }
}