#ifndef MODEL_HPP
#define MODEL_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"
#include <vector>
#include <string>

enum model_type {
	MT_PLAYER,
	MT_STATIC_ASSET,
	MT_LOOTABLE_ITEM
};

struct model {
	unsigned int id;

	enum model_type mt;

	float model_scale;
	unsigned int model_positions;

	struct vector3<unsigned int> model_dimensions;
	unsigned int model_med_positions;
	unsigned int model_lo_positions;


	float shadow_scale;
	struct vector3<unsigned int> shadow_dimensions;
	struct vector2<unsigned int> shadow_offset;
	unsigned int shadow_positions;
	unsigned int shadow_med_positions;
	unsigned int shadow_lo_positions;
};

using namespace std;

void model_init(struct model* m, struct bit_field* bf_assets, const vector<string> file_prefixes, const unsigned int id, const enum model_type mt, const unsigned int i_from, const unsigned int i_to, const float model_scale, const struct vector3<unsigned int> model_dimensions, const float shadow_scale, const struct vector3<unsigned int> shadow_dimensions, const struct vector2<unsigned int> shadow_offset);
struct vector3<float> model_get_max_position(struct model* m);

#endif