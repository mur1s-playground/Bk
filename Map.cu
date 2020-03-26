#include "Map.hpp"

#include "AssetLoader.hpp"

#include "Util.hpp"
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

#include "stdio.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include "Main.hpp"
#include "Grid.hpp"
#include "Map.hpp"
#include "Entity.hpp"
#include "UI.hpp"
#include "MapEditor.hpp"
#include "AssetList.hpp"

struct game_map gm;

unsigned int map_models_position;
vector<struct model> map_models;

unsigned int map_list_pos;

vector<pair<string, string>> map_static_assets;

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

__global__ void draw_map_kernel(const unsigned int* device_data, const struct vector2<unsigned int> map_dimensions_center, const unsigned int map_zoom_level_center, const unsigned int map_zoom_level,
    const unsigned int map_zoom_level_offsets_position, const unsigned int map_positions,
    const unsigned int width, const unsigned int height, const unsigned int channels,
    const unsigned int crop_x1, const unsigned int crop_x2, const unsigned int crop_y1, const unsigned int crop_y2,
    unsigned int* device_data_output, const unsigned int frame_position_target,
    const unsigned int width_target, const unsigned int height_target,
    const float sampling_filter_width_ratio, const unsigned int sampling_filter_width, const float sampling_filter_height_ratio, const unsigned int sampling_filter_height
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width_target * height_target) {
        int current_x = (i % width_target);
        int current_y = (i / width_target);

        unsigned char* target_frame = (unsigned char*)&device_data_output[frame_position_target];

        float current_tile_x_max = map_dimensions_center[0];
        float current_tile_y_max = map_dimensions_center[1];
        float zero_fac = 1.0f;
        if (map_zoom_level < map_zoom_level_center) {
            current_tile_x_max /= 2.0f;
            current_tile_y_max /= 2.0f;
            zero_fac *= 2.0f;
        }
        else if (map_zoom_level > map_zoom_level_center) {
            current_tile_x_max *= 2.0f;
            current_tile_y_max *= 2.0f;
            zero_fac /= 2.0f;
        }
        int last_width = (int)current_tile_x_max % 1920;
        int last_height = (int)current_tile_y_max % 1080;
        current_tile_x_max /= 1920.0f;
        current_tile_y_max /= 1080.0f;

        int current_tile_x_max_i = (int)ceilf(current_tile_x_max);
        int current_tile_y_max_i = (int)ceilf(current_tile_y_max);

        float current_source_x = crop_x1 + (current_x * sampling_filter_width_ratio);
        float current_source_y = crop_y1 + (current_y * sampling_filter_height_ratio);

        int current_source_x_i = (int)floorf(current_source_x);
        int current_source_y_i = (int)floorf(current_source_y);       

        int current_tile_x = current_source_x_i / (1920 * zero_fac);
        int current_tile_y = current_source_y_i / (1080 * zero_fac);

        float current_tile_source_x = current_source_x/zero_fac - 1920 * current_tile_x;
        float current_tile_source_y = current_source_y/zero_fac - 1080 * current_tile_y;

        int current_tile_width = width;
        int current_tile_height = height;
        if (current_tile_x == current_tile_x_max_i - 1) {
            if (last_width == 0) last_width = 1920;
            current_tile_width = last_width;
        }
        if (current_tile_y == current_tile_y_max_i - 1) {
            if (last_height == 0) last_height = 1080;
            current_tile_height = last_height;
        }

        unsigned int zoom_level_offset = device_data[map_zoom_level_offsets_position + map_zoom_level];
        unsigned int map_position = device_data[map_positions + zoom_level_offset + current_tile_y * current_tile_x_max_i + current_tile_x];

        unsigned char* frame = (unsigned char*)&device_data[map_position];

        float components[3];
        float value[3];
        for (int c = 0; c < channels; c++) {
            components[c] = 0.0f;
            value[c] = 0.0f;
        }
        for (int y = 0; y < sampling_filter_height; y++) {
            for (int x = 0; x < sampling_filter_width; x++) {
                if (current_tile_source_y + y < current_tile_height && current_tile_source_x + x < current_tile_width) {
                    for (int c = 0; c < channels; c++) {
                        value[c] += getInterpixel(frame, current_tile_width, current_tile_height, channels, current_tile_source_x + x, current_tile_source_y + y, c);
                        components[c] += 1.0f;
                    }
                }
            }
        }
        for (int c = 0; c < channels; c++) {
            target_frame[current_y * (width_target * channels) + current_x * channels + c] = (unsigned char)roundf(value[c] / components[c]);
        }
    }
}

void launch_draw_map(const unsigned int* device_data, const unsigned int map_zoom_level_count, const unsigned int map_zoom_center_z,
    const unsigned int map_zoom_level_offsets_position, const unsigned int map_positions,
    const unsigned int width, const unsigned int height, const unsigned int channels,
    const unsigned int crop_x1, const unsigned int crop_x2, const unsigned int crop_y1, const unsigned int crop_y2,
    unsigned int* device_data_output, const unsigned int frame_position_target,
    const unsigned int width_target, const unsigned int height_target) {

    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (width_target * height_target + threadsPerBlock - 1) / threadsPerBlock;

    int zoom_level = map_zoom_center_z;
    float sampling_filter_width_ratio = (crop_x2 - crop_x1) / (float)(width_target);
    float sampling_filter_height_ratio = (crop_y2 - crop_y1) / (float)(height_target);

    float tmp_sfw = sampling_filter_width_ratio;

    if (sampling_filter_width_ratio >= 2) {
        //zoom out
        while (sampling_filter_width_ratio >= 2 && zoom_level > 0) {
            sampling_filter_width_ratio /= 2.0f;
            //sampling_filter_height_ratio /= 2.0f;
            zoom_level--;
        }
    } else if (sampling_filter_width_ratio < 0.5) {
        //zoom in
        while (sampling_filter_width_ratio < 0.5 && zoom_level < map_zoom_level_count - 1) {
            sampling_filter_width_ratio *= 2.0f;
            //sampling_filter_height_ratio *= 2.0f;
            zoom_level++;
        }
    }

    int sampling_filter_width = (int)ceilf(sampling_filter_width_ratio);
    int sampling_filter_height = (int)ceilf(sampling_filter_height_ratio);

    sampling_filter_width_ratio = tmp_sfw;

    draw_map_kernel << <blocksPerGrid, threadsPerBlock >> > (device_data, gm.map_dimensions, map_zoom_center_z, zoom_level, map_zoom_level_offsets_position, map_positions, width, height, channels, crop_x1, crop_x2, crop_y1, crop_y2, device_data_output, frame_position_target, width_target, height_target, sampling_filter_width_ratio, sampling_filter_width, sampling_filter_height_ratio, sampling_filter_height);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in draw_map_kernel (error code %s)\n", cudaGetErrorString(err));
    }
}

void map_list_init(struct bit_field *bf_rw) {
    vector<string> maps_cfgs = get_all_files_names_within_folder("./maps", "*", "cfg");

    int size = maps_cfgs.size() * sizeof(struct maplist_element);
    int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));

    map_list_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;

    struct maplist_element* me = (struct maplist_element*) &bf_rw->data[map_list_pos];

    int map_count = 0;
    for (int i = 0; i < maps_cfgs.size(); i++) {
        size_t pos = maps_cfgs[i].find_last_of(".");
        if (pos != string::npos) {
            string map_name_prefix = maps_cfgs[i].substr(0, pos);
            asset_loader_load_file(&bf_assets, "./maps/", map_name_prefix + "_minimap.png", 4);
            struct maplist_element mes(map_name_prefix.c_str());
            memcpy(&me[map_count++], &mes, sizeof(struct maplist_element));
        }
    }

    ui_value_as_config(bf_rw, "lobby", "maps", 0, map_list_pos);
    ui_value_as_config(bf_rw, "lobby", "maps", 1, map_count);
    bit_field_update_device(bf_rw, 0);
}

string map_name_from_index(struct bit_field *bf_rw, const unsigned int idx) {
    struct maplist_element* me = (struct maplist_element*) &bf_rw->data[map_list_pos];
    string name = "";
    for (int i = 0; i < 52; i++) {
        if (me[idx].value[i] == '\0') break;
        name += me[idx].value[i];
    }
    return name;
}

void map_load(struct bit_field *bf_assets, string name) {
    vector<pair<string, string>> map_cfg = get_cfg_key_value_pairs("./maps", name + ".cfg");
    vector<pair<string, string>>::iterator it = map_cfg.begin();
    gm.bf_map = bf_assets;
    while (it != map_cfg.end()) {
        if (it->first == "dimensions") {
            size_t sep = it->second.find(",");
            if (sep != string::npos) {
                string first = trim(it->second.substr(0, sep));
                string second = trim(it->second.substr(sep + 1));
                gm.map_dimensions = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
            }
        }
        if (it->first == "zoom_center_z") {
            gm.map_zoom_center_z = (unsigned int)stoi(it->second);
        }
        if (it->first == "zoom_level_count") {
            gm.map_zoom_level_count = (unsigned int)stoi(it->second);
        }
        it++;
    }

    if (map_editor_update_assets) {
        bit_field_init(bf_assets, 16, 1024);
        bit_field_register_device(bf_assets, 0);

        asset_loader_load_map(bf_assets, name, name + "_", 4);
        asset_loader_load_map(bf_assets, name, name + ".", 1);

        vector<unsigned int> map_positions;
        vector<unsigned int> map_zoom_level_offsets;

        map_zoom_level_offsets.emplace_back(0);
        for (int i = 0; i < gm.map_zoom_level_count; i++) {
            stringstream ss;
            ss << i;
            float tiledim_x = gm.map_dimensions[0] / 1920.0f;
            float tiledim_y = gm.map_dimensions[1] / 1080.0f;
            for (int x = i; i < gm.map_zoom_center_z; x++) {
                tiledim_x /= 2.0f;
                tiledim_y /= 2.0f;
            }
            for (int x = 0; i > gm.map_zoom_center_z&& i < gm.map_zoom_level_count && x < i; x++) {
                tiledim_x *= 2.0f;
                tiledim_y *= 2.0f;
            }
            unsigned int td_x = (unsigned int)ceilf(tiledim_x);
            unsigned int td_y = (unsigned int)ceilf(tiledim_y);
            map_zoom_level_offsets.emplace_back(td_x * td_y + map_zoom_level_offsets[i]);
            printf("tiledim_x %i, tiledim_y %i\n", td_x, td_y);
            for (int y = 0; y < tiledim_y; y++) {
                stringstream sy;
                sy << y;
                for (int x = 0; x < tiledim_x; x++) {
                    stringstream sx;
                    sx << x;
                    map_positions.emplace_back(assets["./maps/" + name + "/" + name + "_" + ss.str() + "_" + sy.str() + "_" + sx.str() + ".png"]);
                    printf("map_init_positions: %i\n", assets["./maps/" + name + "/" + name + "_" + ss.str() + "_" + sy.str() + "_" + sx.str() + ".png"]);
                }
            }
        }
        gm.map_zoom_level_offsets_position = bit_field_add_bulk(bf_assets, map_zoom_level_offsets.data(), map_zoom_level_offsets.size(), map_zoom_level_offsets.size() * sizeof(unsigned int)) + 1;
        gm.map_positions = bit_field_add_bulk(bf_assets, map_positions.data(), map_positions.size(), map_positions.size() * sizeof(unsigned int)) + 1;
        gm.map_loot_probabilities_position = assets["./maps/" + name + "/" + name + ".loot_probabilities.png"];
        gm.map_spawn_probabilities_position = assets["./maps/" + name + "/" + name + ".spawn_probabilities.png"];
        gm.map_pathable_position = assets["./maps/" + name + "/" + name + ".pathable.png"];

        map_static_assets = get_cfg_key_value_pairs("./maps/" + name + "/", name + "_static_assets.cfg");

        //load map assets
        vector<string> model_cfgs = get_all_files_names_within_folder("./maps/" + name + "/assets/", "*", "cfg");
        vector<struct model> tmp_models;
        for (int i = 0; i < model_cfgs.size(); i++) {
            struct model m = model_from_cfg(bf_assets, "./maps/" + name + "/assets/", model_cfgs[i]);
            if (map_editor) {
                size_t dot_pos = model_cfgs[i].find_last_of('.');
                if (dot_pos != string::npos) {
                    string m_name = model_cfgs[i].substr(0, dot_pos);
                    assetlist_add(&bf_rw, m.id, m_name.c_str());
                }
            }
            tmp_models.push_back(m);
        }
        int counter = 0;
        struct model empty_model;
        empty_model.id = UINT_MAX;
        while (counter < tmp_models.size()) {
            for (int i = 0; i < tmp_models.size(); i++) {
                if (tmp_models[i].id == counter + 100) {
                    map_models.push_back(tmp_models[i]);
                }
            }
            if (map_models.size() < counter + 1) {
                map_models.push_back(empty_model);
            }
            counter++;
        }
        unsigned int size = map_models.size() * sizeof(struct model);
        unsigned int size_in_bf = (unsigned int)ceilf(size / (float)sizeof(unsigned int));
        map_models_position = bit_field_add_bulk(bf_assets, (unsigned int*)map_models.data(), size_in_bf, size) + 1;
    } else {
        if (map_editor) {
            vector<string> model_cfgs = get_all_files_names_within_folder("./maps/" + name + "/assets/", "*", "cfg");
            for (int i = 0; i < model_cfgs.size(); i++) {
                struct model m = model_from_cfg(bf_assets, "./maps/" + name + "/assets/", model_cfgs[i], true);
                if (map_editor) {
                    size_t dot_pos = model_cfgs[i].find_last_of('.');
                    if (dot_pos != string::npos) {
                        string m_name = model_cfgs[i].substr(0, dot_pos);
                        assetlist_add(&bf_rw, m.id, m_name.c_str());
                    }
                }
            }
        }

        vector<pair<string, string>> map_bf_txt = get_cfg_key_value_pairs("./maps/" + name + "/", name + ".bf.txt");
        for (int i = 0; i < map_bf_txt.size(); i++) {
            string cfg_val = map_bf_txt[i].first;
            cfg_val = trim(cfg_val);
            if (cfg_val == "map_zoom_level_offsets_position") {
                string second = map_bf_txt[i].second;
                second = trim(second);
                gm.map_zoom_level_offsets_position = stoi(second);
            }
            if (cfg_val == "map_loot_probabilities_position") {
                string second = map_bf_txt[i].second;
                second = trim(second);
                gm.map_loot_probabilities_position = stoi(second);
            }
            if (cfg_val == "map_pathable_position") {
                string second = map_bf_txt[i].second;
                second = trim(second);
                gm.map_pathable_position = stoi(second);
            }
            if (cfg_val == "map_positions") {
                string second = map_bf_txt[i].second;
                second = trim(second);
                gm.map_positions = stoi(second);
            }
            if (cfg_val == "map_spawn_probabilities_position") {
                string second = map_bf_txt[i].second;
                second = trim(second);
                gm.map_spawn_probabilities_position = stoi(second);
            }
            if (cfg_val == "map_models_position") {
                string second = map_bf_txt[i].second;
                second = trim(second);
                map_models_position = stoi(second);
            }
        }

        map_static_assets = get_cfg_key_value_pairs("./maps/" + name + "/", name + "_static_assets.cfg");
        bit_field_load_from_disk(bf_assets, "./maps/" + name + "/" + name + ".bf");
        bit_field_register_device(bf_assets, 0);
    }
    bit_field_update_device(bf_assets, 0);
}

void map_add_static_assets(struct bit_field* bf_assets, struct bit_field* bf_grid, struct grid* gd) {
    printf("starting static asset addition\n");

    for (int i = 0; i < map_static_assets.size(); i++) {
        string a_id_str = map_static_assets[i].first;
        a_id_str = trim(a_id_str);

        unsigned int a_id = stoi(a_id_str);

        string a_str = map_static_assets[i].second;
        size_t sep_pos = a_str.find_first_of(':', 0);

        string a_coords = a_str.substr(0, sep_pos);

        size_t a_coords_sep = a_coords.find(",");
        string a_coords_x = a_coords.substr(0, a_coords_sep);
        a_coords_x = trim(a_coords_x);
        string a_coords_y = a_coords.substr(a_coords_sep + 1);
        a_coords_y = trim(a_coords_y);

        size_t sep_pos2 = a_str.find_first_of(':', sep_pos + 1);
        string a_orientation = a_str.substr(sep_pos + 1, sep_pos2-sep_pos);
        a_orientation = trim(a_orientation);

        sep_pos = a_str.find_first_of(':', sep_pos2 + 1);
        string a_scale = a_str.substr(sep_pos2+1, sep_pos-sep_pos2);
        a_scale = trim(a_scale);

        sep_pos2 = a_str.find_first_of(':', sep_pos + 1);
        string a_zindex = a_str.substr(sep_pos+1, sep_pos2-sep_pos);
        a_zindex = trim(a_zindex);

        string a_aoffset = a_str.substr(sep_pos2 + 1);
        a_aoffset = trim(a_aoffset);

        printf("adding asset: %i at x %i y %i, orientation %i, scale: %f, z %i, a_offset: %i\n", a_id, stoi(a_coords_x), stoi(a_coords_y), stoi(a_orientation), stof(a_scale), stoi(a_zindex), stoi(a_aoffset));

        struct model* map_models_bfs = (struct model *) &bf_map.data[map_models_position];

        entity_add("static_asset", ET_STATIC_ASSET, a_id, stoi(a_zindex));
        struct entity* cur_e = &entities[entities.size() - 1];
        cur_e->position = { (float)stoi(a_coords_x), (float)stoi(a_coords_y), 0.0f };
        cur_e->scale = stof(a_scale);
        cur_e->orientation = stoi(a_orientation);
        cur_e->model_animation_offset = stoi(a_aoffset);

        struct vector3<float> max_pos = model_get_max_position(&map_models_bfs[a_id - 100]) * map_models_bfs[a_id - 100].model_scale;
        grid_object_add(bf_grid, bf_grid->data, gd->position_in_bf, cur_e->position, { stof(a_scale), stof(a_scale), stof(a_scale) }, { 0.0f, 0.0f, 0.0f }, max_pos, entities.size() - 1);
    }
    printf("finished static asset addition\n");
}