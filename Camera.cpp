#include "Camera.hpp"

#include "Main.hpp"
#include "Map.hpp"

struct vector3<float>	camera;

vector<unsigned int> camera_crop;
vector<unsigned int> camera_crop_tmp;

float					camera_z_from			= 1.0f;
float					camera_z_target			= 1.0f;
unsigned int			camera_z_tick_target	= 30;
unsigned int			camera_z_tick_counter	= 0;

void camera_init() {
	camera_crop.push_back(0);
	camera_crop.push_back(0);
	camera_crop.push_back(0);
	camera_crop.push_back(0);

	camera_crop_tmp.push_back(0);
	camera_crop_tmp.push_back(0);
	camera_crop_tmp.push_back(0);
	camera_crop_tmp.push_back(0);
}

void camera_move_z(float z_delta) {
	camera_z_target += z_delta;
	camera_z_from = camera[2];
	camera_z_tick_counter = camera_z_tick_target;
}

void camera_move_z_tick() {
	if (camera_z_tick_counter > 0) {
		camera_z_tick_counter--;
		float camera_z = camera[2];
		camera_crop_tmp = camera_crop;

		float delta_z = (camera_z_target - camera_z_from);
		float delta_z_dist = abs(delta_z);
		if (delta_z_dist < 0.5) {
			camera_z_tick_target = 5;
		} else if (delta_z_dist < 1){
			camera_z_tick_target = 15;
		} else {
			camera_z_tick_target = 30;
		}
		camera_z_tick_counter = min(camera_z_tick_target-1, camera_z_tick_counter);

		float z_interp = camera_z_from + (camera_z_tick_target - camera_z_tick_counter)/((float) camera_z_tick_target) * delta_z;
		if (delta_z < 0) {
			if (z_interp >= 0.2f) {
				camera[2] = z_interp;
			} else {
				camera_z_target = camera[2];
				camera_z_tick_counter = 0;
			}
		}
		if (z_interp >= 0.2f) {
			bool delta_z_happens = true;
			if (camera[0] + resolution[0] * (z_interp) >= gm.map_dimensions[0]) {
				if (resolution[0] * (z_interp) < gm.map_dimensions[0] && resolution[1] * (z_interp) < gm.map_dimensions[1]) {
					camera[0] = gm.map_dimensions[0] - (unsigned int)ceilf(resolution[0] * (z_interp)) - 1;
				} else {
					delta_z_happens = false;
				}
			}
			if (camera[1] + resolution[1] * (z_interp) >= gm.map_dimensions[1]) {
				if (resolution[0] * (z_interp) < gm.map_dimensions[0] && resolution[1] * (z_interp) < gm.map_dimensions[1]) {
					camera[1] = gm.map_dimensions[1] - (unsigned int)ceilf(resolution[1] * (z_interp)) - 1;
				} else {
					delta_z_happens = false;
				}
			}
			if (delta_z_happens) {
				camera[2] = z_interp;
			} else {
				camera_z_target = camera[2];
				camera_z_tick_counter = 0;
			}
		}
		camera_get_crop(camera_crop);
		if (camera_z != camera[2]) {
			camera_move(struct vector3<float>(camera_crop_tmp[0] - camera_crop[0] + mouse_position[0] * (camera_z - camera[2]), 0.0f, 0.0f));
			camera_get_crop(camera_crop);
			camera_move(struct vector3<float>(0.0f, camera_crop_tmp[2] - camera_crop[2] + mouse_position[1] * (camera_z - camera[2]), 0.0f));
			camera_get_crop(camera_crop);
		}
	} else {
		camera_z_tick_counter = 0;
	}
}

void camera_move(struct vector3<float> delta) {
	if (delta[2] != 0) {
		camera_move_z(delta[2]);
	} else {
		if (camera[0] + delta[0] + resolution[0] * camera[2] >= gm.map_dimensions[0]) {
			camera[0] = gm.map_dimensions[0] - (unsigned int)ceilf(resolution[0] * camera[2]) - 1;
		}
		else {
			camera[0] += delta[0];
		}
		if (camera[0] + delta[0] < 0) {
			camera[0] = 0;
		}
		if (camera[1] + delta[1] + resolution[1] * camera[2] >= gm.map_dimensions[1]) {
			camera[1] = gm.map_dimensions[1] - (unsigned int)ceil(resolution[1] * camera[2]) - 1;
		}
		else {
			camera[1] += delta[1];
		}
		if (camera[1] + delta[1] < 0) {
			camera[1] = 0;
		}
	}
}

void camera_get_crop(vector<unsigned int> &out_crop) {
	out_crop[0] = camera[0];
	out_crop[1] = camera[0] + (unsigned int)floorf(resolution[0] * camera[2]);
	out_crop[2] = camera[1];
	out_crop[3] = camera[1] + (unsigned int)floorf(resolution[1] * camera[2]);
}