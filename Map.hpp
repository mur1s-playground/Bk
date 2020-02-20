#ifndef MAP_HPP
#define MAP_HPP

#include "BitField.hpp"
#include "Vector2.hpp"

#include <string>
#include <vector>

struct game_map {
	struct bit_field*				bf_map;
	struct vector2<unsigned int>	map_dimensions;
	unsigned int					map_zoom_center_z;
	
	unsigned int					map_zoom_level_count;

	unsigned int					map_zoom_level_offsets_position;
	unsigned int					map_positions;

	unsigned int					map_static_assets_position;
	unsigned int					map_loot_probabilities_position;
	unsigned int					map_spawn_probabilities_position;
	unsigned int					map_pathable_position;

	unsigned int					map_storm_pathing;
};



extern struct game_map gm;

using namespace std;

extern vector<string> game_maps;

extern unsigned int map_models_position; 
extern vector<struct model> map_models;

void launch_draw_map(const unsigned int* device_data, const unsigned int map_zoom_level_count, const unsigned int map_zoom_center_z,
	const unsigned int map_zoom_level_offsets_position, const unsigned int map_positions,
	const unsigned int width, const unsigned int height, const unsigned int channels,
	const unsigned int crop_x1, const unsigned int crop_x2, const unsigned int crop_y1, const unsigned int crop_y2,
	unsigned int* device_data_output, const unsigned int frame_position_target,
	const unsigned int width_target, const unsigned int height_target);

void map_list_init();
void map_load(struct bit_field* bf_assets, string name);

void map_add_static_assets(struct bit_field* bf_assets, struct bit_field* bf_grid, struct grid* gd);

#endif