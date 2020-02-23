#ifndef STORM_HPP
#define STORM_HPP

#include "Vector3.hpp"
#include <vector>

using namespace std;

struct storm {
	unsigned int	x;
	unsigned int	y;
	float			radius;
};

extern unsigned int			storm_phase_current;

extern vector<unsigned int> storm_phase_start_ticks;
extern vector<unsigned int> storm_phase_duration_ticks;
extern vector<float>		storm_phase_mapratio;
extern vector<float>		storm_phase_dps;

extern struct storm storm_current;
extern struct storm storm_to;

void storm_init(struct bit_field* bf_map, struct bit_field* bf_rw);
void storm_next(struct bit_field* bf_map, struct bit_field* bf_rw);
bool storm_is_in(vector3<float> position);

void launch_draw_storm_kernel(unsigned int* device_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels, const unsigned int camera_crop_x1, const unsigned int camera_crop_y1, const float camera_z, const struct storm storm_current, const struct storm storm_to, const unsigned int storm_alpha, const struct vector3<unsigned char> storm_color);

#endif