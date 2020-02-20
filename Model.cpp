#include "Model.hpp"

#include "AssetLoader.hpp"

#include <sstream>
#include "Util.hpp"

vector<struct model> models;

struct model model_from_cfg(struct bit_field *bf_assets, string folder, string filename) {
    size_t dot_pos = filename.find_last_of('.');
    asset_loader_load_folder(bf_assets, folder + filename.substr(0, dot_pos) + "/");
    map<string, string> cfg_kv = get_cfg_key_value_pairs(folder, filename);
    map<string, string>::iterator it = cfg_kv.begin();
    struct model m;
    vector<unsigned int> model_assets_positions;
    vector<unsigned int> shadow_assets_positions;
    while (it != cfg_kv.end()) {
        if (it->first == "id") {
            m.id = (unsigned int)stoi(it->second);
        }
        if (it->first == "type") {
            if (it->second == "PLAYER") {
                m.mt = MT_PLAYER;
            } else if (it->second == "LOOTABLE_ITEM") {
                m.mt = MT_LOOTABLE_ITEM;
            } else if (it->second == "STATIC_ASSET") {
                m.mt = MT_STATIC_ASSET;
            }
        }
        if (it->first == "model_scale") {
            m.model_scale = stof(it->second);
        }
        if (it->first == "model_zoom_level_count") {
            m.model_zoom_level_count = stoi(it->second);
        }
        if (it->first == "model_dimensions") {
            size_t sep = it->second.find(",");
            if (sep != string::npos) {
                string first = trim(it->second.substr(0, sep));
                string second = trim(it->second.substr(sep + 1));
                m.model_dimensions = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
            }
        }
        if (it->first == "model_animation_ticks") {
            m.model_animation_ticks = stoi(it->second);
        }
        if (it->first == "model_animation_stepsize") {
            m.model_animation_stepsize = stoi(it->second);
        }
        for (int mz = 0; mz < m.model_zoom_level_count; mz++) {
            stringstream ss;
            ss << mz;
            if (it->first == "model_" + ss.str() + "_file_prefix") {
                vector<unsigned int> model_assets_positions_mz;
                for (int r = 1; r <= 36 * m.model_animation_ticks; r++) {
                    stringstream sr;
                    if (r < 10) {
                        sr << 0;
                    }
                    sr << r;
                    model_assets_positions_mz.push_back(assets[folder + filename.substr(0, dot_pos) + "/" + it->second + sr.str() + ".png"]);
                }
                model_assets_positions.push_back(bit_field_add_bulk(bf_assets, model_assets_positions_mz.data(), model_assets_positions_mz.size(), model_assets_positions_mz.size() * sizeof(unsigned int)) + 1);
            }
        }

        if (it->first == "shadow_scale") {
            m.model_scale = stof(it->second);
        }
        if (it->first == "shadow_zoom_level_count") {
            m.model_zoom_level_count = stoi(it->second);
        }
        if (it->first == "shadow_dimensions") {
            size_t sep = it->second.find(",");
            if (sep != string::npos) {
                string first = trim(it->second.substr(0, sep));
                string second = trim(it->second.substr(sep + 1));
                m.model_dimensions = { (unsigned int)stoi(first), (unsigned int)stoi(second) };
            }
        }
        if (it->first == "shadow_animation_ticks") {
            m.model_animation_ticks = stoi(it->second);
        }
        if (it->first == "shadow_animation_stepsize") {
            m.model_animation_stepsize = stoi(it->second);
        }
        for (int mz = 0; mz < m.shadow_zoom_level_count; mz++) {
            stringstream ss;
            ss << mz;
            if (it->first == "shadow_" + ss.str() + "_file_prefix") {
                vector<unsigned int> model_shadow_assets_positions_mz;
                for (int r = 1; r <= 36 * m.shadow_animation_ticks; r++) {
                    stringstream sr;
                    if (r < 10) {
                        sr << 0;
                    }
                    sr << r;
                    model_shadow_assets_positions_mz.push_back(assets[folder + filename.substr(0, dot_pos) + "/" + it->second + sr.str() + ".png"]);
                }
                shadow_assets_positions.push_back(bit_field_add_bulk(bf_assets, model_shadow_assets_positions_mz.data(), model_shadow_assets_positions_mz.size(), model_shadow_assets_positions_mz.size() * sizeof(unsigned int)) + 1);
            }
        }
        it++;
    }
    m.model_positions = bit_field_add_bulk(bf_assets, model_assets_positions.data(), model_assets_positions.size(), model_assets_positions.size() * sizeof(unsigned int)) + 1;
    m.shadow_positions = bit_field_add_bulk(bf_assets, shadow_assets_positions.data(), shadow_assets_positions.size(), shadow_assets_positions.size() * sizeof(unsigned int)) + 1;
    return m;
}
/*
void model_init(struct model *m, struct bit_field *bf_assets, const vector<string> file_prefixes, const unsigned int id, const enum model_type mt, const unsigned int i_from, const unsigned int i_to, const float model_scale, const struct vector3<unsigned int> model_dimensions, const float shadow_scale, const struct vector3<unsigned int> shadow_dimensions, const struct vector2<unsigned int> shadow_offset) {
	vector<unsigned int> model_positions;
	vector<unsigned int> model_med_positions;
	vector<unsigned int> model_lo_positions;
	vector<unsigned int> shadow_positions;
	vector<unsigned int> shadow_med_positions;
	vector<unsigned int> shadow_lo_positions;

	for (int i = i_from; i <= i_to; i++) {
		stringstream ls;
		if (i < 10) ls << 0;
		ls << i;
		model_positions.push_back(assets[file_prefixes[0] + ls.str() + ".png"]);
		model_med_positions.push_back(assets[file_prefixes[1] + ls.str() + ".png"]);
		model_lo_positions.push_back(assets[file_prefixes[2] + ls.str() + ".png"]);
		shadow_positions.push_back(assets[file_prefixes[3] + ls.str() + ".png"]);
		shadow_med_positions.push_back(assets[file_prefixes[4] + ls.str() + ".png"]);
		shadow_lo_positions.push_back(assets[file_prefixes[5] + ls.str() + ".png"]);
	}

	m->id = id;
	m->mt = mt;
	m->model_scale = model_scale;
	m->model_dimensions = model_dimensions;
	m->model_positions = bit_field_add_bulk(bf_assets, model_positions.data(), 1+i_to-i_from, (1 + i_to - i_from) * sizeof(unsigned int)) + 1;
	m->model_med_positions = bit_field_add_bulk(bf_assets, model_med_positions.data(), 1 + i_to - i_from, (1 + i_to - i_from) * sizeof(unsigned int)) + 1;
	m->model_lo_positions = bit_field_add_bulk(bf_assets, model_lo_positions.data(), 1 + i_to - i_from, (1 + i_to - i_from) * sizeof(unsigned int)) + 1;
	m->shadow_scale = shadow_scale;
	m->shadow_dimensions = shadow_dimensions;
	m->shadow_offset = shadow_offset;
	m->shadow_positions = bit_field_add_bulk(bf_assets, shadow_positions.data(), 1 + i_to - i_from, (1 + i_to - i_from) * sizeof(unsigned int)) + 1;
	m->shadow_med_positions = bit_field_add_bulk(bf_assets, shadow_med_positions.data(), 1 + i_to - i_from, (1 + i_to - i_from) * sizeof(unsigned int)) + 1;
	m->shadow_lo_positions = bit_field_add_bulk(bf_assets, shadow_lo_positions.data(), 1 + i_to - i_from, (1 + i_to - i_from) * sizeof(unsigned int)) + 1;
}
*/

struct vector3<float> model_get_max_position(struct model* m) {
	vector3<float> max_position = { 
		max((float)((m->shadow_offset[0] + m->shadow_dimensions[0])), (float)(m->model_dimensions[0])) + 32, 
		max((float)((m->shadow_offset[1] + m->shadow_dimensions[1])), (float)(m->model_dimensions[1])) + 32,
		0.0f
	};
	return max_position;
}