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
	unsigned int					id;

	enum model_type					mt;

	float							model_scale;
	unsigned int					model_zoom_level_count;
	struct vector2<unsigned int>	model_dimensions;
	unsigned int					model_animation_ticks;
	unsigned int					model_animation_stepsize;

	unsigned int					model_positions;

	float							shadow_scale;
	struct vector2<unsigned int>	shadow_offset;
	unsigned int					shadow_zoom_level_count;
	struct vector2<unsigned int>	shadow_dimensions;
	unsigned int					shadow_animation_ticks;
	unsigned int					shadow_animation_stepsize;
	
	unsigned int					shadow_positions;
};

using namespace std;

extern vector<struct model> models;

struct model model_from_cfg(struct bit_field* bf_assets, string folder, string filename);
//void model_init(struct model* m, struct bit_field* bf_assets, const vector<string> file_prefixes, const unsigned int id, const enum model_type mt, const unsigned int i_from, const unsigned int i_to, const float model_scale, const struct vector3<unsigned int> model_dimensions, const float shadow_scale, const struct vector3<unsigned int> shadow_dimensions, const struct vector2<unsigned int> shadow_offset);
struct vector3<float> model_get_max_position(struct model* m);

#endif