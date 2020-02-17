#include "Model.hpp"

#include "AssetLoader.hpp"

#include <sstream>

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

struct vector3<float> model_get_max_position(struct model* m) {
	vector3<float> max_position = { 
		max((float)((m->shadow_offset[0] + m->shadow_dimensions[0])), (float)(m->model_dimensions[0])) + 32, 
		max((float)((m->shadow_offset[1] + m->shadow_dimensions[1])), (float)(m->model_dimensions[1])) + 32,
		0.0f
	};
	return max_position;
}