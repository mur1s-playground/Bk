#include "Model.hpp"

#include "AssetLoader.hpp"

#include <sstream>
#include "Util.hpp"
#include "BitField.hpp"

vector<struct model> models;

struct model model_from_cfg(struct bit_field *bf_assets, string folder, string filename) {
    size_t dot_pos = filename.find_last_of('.');
    asset_loader_load_folder(bf_assets, folder + filename.substr(0, dot_pos) + "/");
    vector<pair<string, string>> cfg_kv = get_cfg_key_value_pairs(folder, filename);
    struct model m;
    m.model_zoom_level_count = 0;
    m.shadow_zoom_level_count = 0;
    vector<unsigned int> model_assets_positions;
    vector<unsigned int> shadow_assets_positions;
    for (int c_i = 0; c_i < cfg_kv.size(); c_i++) {
        if (cfg_kv[c_i].first == "id") {
            m.id = (unsigned int)stoi(cfg_kv[c_i].second);
            printf("model.id %u\n", m.id);
        }
        if (cfg_kv[c_i].first == "type") {
            printf("model.type ");
            if (cfg_kv[c_i].second == "PLAYER") {
                m.mt = MT_PLAYER;
                printf("player\n");
            } else if (cfg_kv[c_i].second == "LOOTABLE_ITEM") {
                m.mt = MT_LOOTABLE_ITEM;
                printf("lootable_item\n");
            } else if (cfg_kv[c_i].second == "STATIC_ASSET") {
                m.mt = MT_STATIC_ASSET;
                printf("static_asset\n");
            }
        }
        if (cfg_kv[c_i].first == "model_scale") {
            m.model_scale = stof(cfg_kv[c_i].second);
            printf("model.scale %f\n", m.model_scale);
        }
        if (cfg_kv[c_i].first == "model_zoom_level_count") {
            m.model_zoom_level_count = stoi(cfg_kv[c_i].second);
            printf("model_zoom_level_count %u\n", m.model_zoom_level_count);
        }
        if (cfg_kv[c_i].first == "model_dimensions") {
            size_t sep = cfg_kv[c_i].second.find(",");
            if (sep != string::npos) {
                string first = cfg_kv[c_i].second.substr(0, sep);
                first = trim(first);
                string second = cfg_kv[c_i].second.substr(sep + 1);
                second = trim(second);
                m.model_dimensions = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                printf("model_dims: %u %u\n", m.model_dimensions[0], m.model_dimensions[1]);
            }
        }
        if (cfg_kv[c_i].first == "model_animation_ticks") {
            m.model_animation_ticks = stoi(cfg_kv[c_i].second);
            printf("model_animation_ticks %u\n", m.model_animation_ticks);
        }
        if (cfg_kv[c_i].first == "model_animation_stepsize") {
            m.model_animation_stepsize = stoi(cfg_kv[c_i].second);
            printf("model_animation_stepsize %u\n", m.model_animation_stepsize);
        }
        for (int mz = 0; mz < m.model_zoom_level_count; mz++) {
            stringstream ss;
            ss << mz;
            if (cfg_kv[c_i].first == "model_" + ss.str() + "_file_prefix") {
                vector<unsigned int> model_assets_positions_mz;
                for (int r = 1; r <= 36 * m.model_animation_ticks; r++) {
                    stringstream sr;
                    if (r < 10) {
                        sr << 0;
                    }
                    sr << r;
                    model_assets_positions_mz.push_back(assets[folder + filename.substr(0, dot_pos) + "/" + cfg_kv[c_i].second + sr.str() + ".png"]);
                }
                model_assets_positions.push_back(bit_field_add_bulk(bf_assets, model_assets_positions_mz.data(), model_assets_positions_mz.size(), model_assets_positions_mz.size() * sizeof(unsigned int)) + 1);
            }
        }
        if (cfg_kv[c_i].first == "shadow_scale") {
            m.shadow_scale = stof(cfg_kv[c_i].second);
            printf("shadow.scale %f\n", m.shadow_scale);
        }
        if (cfg_kv[c_i].first == "shadow_offset") {
            size_t sep = cfg_kv[c_i].second.find(",");
            if (sep != string::npos) {
                string first = cfg_kv[c_i].second.substr(0, sep);
                first = trim(first);
                string second = cfg_kv[c_i].second.substr(sep + 1);
                second = trim(second);
                m.shadow_offset = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                printf("shadow_offset: %u %u\n", m.shadow_offset[0], m.shadow_offset[1]);
            }
        }
        if (cfg_kv[c_i].first == "shadow_zoom_level_count") {
            m.shadow_zoom_level_count = stoi(cfg_kv[c_i].second);
            printf("shadow_zoom_level_count %u\n", m.shadow_zoom_level_count);
        }
        if (cfg_kv[c_i].first == "shadow_dimensions") {
            size_t sep = cfg_kv[c_i].second.find(",");
            if (sep != string::npos) {
                string first = cfg_kv[c_i].second.substr(0, sep);
                first = trim(first);
                string second = cfg_kv[c_i].second.substr(sep + 1);
                second = trim(second);
                m.shadow_dimensions = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
                printf("shadow_dims: %u %u\n", m.shadow_dimensions[0], m.shadow_dimensions[1]);
            }
        }
        if (cfg_kv[c_i].first == "shadow_animation_ticks") {
            m.shadow_animation_ticks = stoi(cfg_kv[c_i].second);
            printf("shadow_animation_ticks %u\n", m.shadow_animation_ticks);
        }
        if (cfg_kv[c_i].first == "shadow_animation_stepsize") {
            m.shadow_animation_stepsize = stoi(cfg_kv[c_i].second);
            printf("shadow_animation_stepsize %u\n", m.shadow_animation_stepsize);
        }
        for (int mz = 0; mz < m.shadow_zoom_level_count; mz++) {
            stringstream ss;
            ss << mz;
            if (cfg_kv[c_i].first == "shadow_" + ss.str() + "_file_prefix") {
                vector<unsigned int> model_shadow_assets_positions_mz;
                for (int r = 1; r <= 36 * m.shadow_animation_ticks; r++) {
                    stringstream sr;
                    if (r < 10) {
                        sr << 0;
                    }
                    sr << r;
                    //printf("adding shadow asset: %s%s/%s%s.png %u\n", folder.c_str(), filename.substr(0, dot_pos).c_str(), cfg_kv[c_i].second.c_str(), sr.str().c_str(), assets[folder + filename.substr(0, dot_pos) + "/" + cfg_kv[c_i].second + sr.str() + ".png"]);
                    model_shadow_assets_positions_mz.push_back(assets[folder + filename.substr(0, dot_pos) + "/" + cfg_kv[c_i].second + sr.str() + ".png"]);
                }
                shadow_assets_positions.push_back(bit_field_add_bulk(bf_assets, model_shadow_assets_positions_mz.data(), model_shadow_assets_positions_mz.size(), model_shadow_assets_positions_mz.size() * sizeof(unsigned int)) + 1);
            }
        }
    }
    m.model_positions = bit_field_add_bulk(bf_assets, model_assets_positions.data(), model_assets_positions.size(), model_assets_positions.size() * sizeof(unsigned int)) + 1;
    m.shadow_positions = bit_field_add_bulk(bf_assets, shadow_assets_positions.data(), shadow_assets_positions.size(), shadow_assets_positions.size() * sizeof(unsigned int)) + 1;
    return m;
}

struct vector3<float> model_get_max_position(struct model* m) {
	vector3<float> max_position = { 
		max((float)((m->shadow_offset[0] + m->shadow_dimensions[0])), (float)(m->model_dimensions[0])), 
		max((float)((m->shadow_offset[1] + m->shadow_dimensions[1])), (float)(m->model_dimensions[1])),
		0.0f
	};
	return max_position;
}