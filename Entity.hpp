#ifndef ENTITY_HPP
#define ENTITY_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"
#include <vector>
#include <string>
#include "FeatureToggles.hpp"

enum entity_type {
	ET_PLAYER,
	ET_ITEM,
	ET_STATIC_ASSET,
	ET_TEXT
};

struct entity {
	enum entity_type		et;

	char					name[50];
	unsigned int			name_len;

	struct vector3<float>	position;
	float					scale;
	float					orientation;

	unsigned int			model_id;
	unsigned int			model_z;
	unsigned int			model_animation_offset;

#ifdef PATHING_DEBUG
	char					params[256];
#else
	char					params[50];
#endif
};

using namespace std;

extern unsigned int             entities_size_in_bf;
extern unsigned int				entities_position;
extern vector<struct entity>	entities;

void launch_draw_entities_kernel(const unsigned int* device_data_assets, const unsigned int *device_data_map, const unsigned int players_models_position, const unsigned int item_models_position, const unsigned int map_models_position, const unsigned int fonts_position,
	const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int gd_position_in_bf, const unsigned int gd_data_position_in_bf,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const struct vector2<unsigned int> mouse_position, const unsigned int tick_counter);

void entity_add(string name, enum entity_type et, unsigned int model_id, unsigned int model_z);
void entities_upload(struct bit_field* bf);

#endif