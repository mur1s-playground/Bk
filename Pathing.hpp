#pragma once

#include "FeatureToggles.hpp"

#include "Vector2.hpp"

#ifdef PATHING_DEBUG

#ifdef BRUTE_PATHING
	void launch_draw_pathing_kernel(
		const unsigned int* device_data_assets,
		const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int entities_count,
		unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
		const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter);
#else	

	void launch_draw_gpu_kernel(
		unsigned int* device_data_rw, const unsigned int pathing_position,
		unsigned int* device_data_pathing,
		unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
		const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter);
#endif

#endif

#ifndef BRUTE_PATHING
struct path {
	vector2<float> from;
	vector2<float> to;

	vector2<unsigned int> resolution;
	float zoom_level;

	vector2<unsigned int> pathing_x1y1;
	vector2<unsigned int> pathing_x2y2;

	unsigned int pathing_data;
	
	int path_calc_stage;
};

unsigned int pathing_add(struct bit_field* bf_rw, struct bit_field* bf_pathing);
bool pathing_set(struct bit_field* bf_rw, unsigned int pathing_position, struct vector2<float> from, struct vector2<float> to);
void pathing_get(struct bit_field* bf_rw, unsigned int pathing_position, struct bit_field* bf_pathing, struct bit_field* bf_map, int path_calc_stage);
vector2<float> pathing_get_next(struct bit_field* bf_rw, unsigned int pathing_position, struct bit_field* bf_pathing, struct vector2<float> position);

void launch_calculate_pathing_kernel(
	const struct path *p,
	const unsigned int* device_data_rw, const unsigned int path_position, unsigned int *device_data_pathing, const int path_calc_stage,
	const struct vector2<unsigned int> map_dimensions_center,
	const unsigned int* device_data_map, const unsigned int map_pathables);
#endif