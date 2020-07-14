#pragma once

#include "Vector2.hpp"
#include <vector>
#include "BitField.hpp"

struct particle {
	int max_lifetime_ticks;

	vector2<float> origin;
	vector2<float> target;

	vector2<float> position;
	float speed_per_tick;
};

using namespace std;

extern const int				particles_max;
extern int						particles_max_used;
extern int						particles_count;
extern struct particle          *particles;

extern unsigned int				particles_position;
extern int						particles_size_in_bf;

void launch_draw_particles_kernel(
	const unsigned int* device_data_assets,
	const unsigned int* device_data_rw, const unsigned int particles_position,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter);

void particles_init(struct bit_field* bf);
void particles_tick(struct bit_field* bf);
void particles_clear(struct bit_field* bf);

void particle_add(struct bit_field* bf, int max_lifetime_ticks, vector2<float> origin, vector2<float> target, float speed_per_tick);

