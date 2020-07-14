#pragma once

#include "FeatureToggles.hpp"

#ifdef PATHING_DEBUG

#include "Vector2.hpp"

void launch_draw_pathing_kernel(
	const unsigned int* device_data_assets,
	const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int entities_count,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter);

#endif