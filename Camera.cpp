#include "Camera.hpp"

#include "Main.hpp"
#include "Map.hpp"

void camera_move(struct vector3<float> delta) {
	if (delta[2] < 0 && camera[2] + delta[2] >= 0.01f) {
		camera[2] += delta[2];
	}
	if (camera[2] + delta[2] >= 0.01f) {
		bool delta_z_happens = true;
		if (camera[0] + resolution[0] * (camera[2] + delta[2]) >= gm.map_dimensions[0]) {
			if (resolution[0] * (camera[2] + delta[2]) < gm.map_dimensions[0] && resolution[1] * (camera[2] + delta[2]) < gm.map_dimensions[1]) {
				camera[0] = gm.map_dimensions[0] - (unsigned int)ceilf(resolution[0] * (camera[2] + delta[2])) - 1;
			} else {
				delta_z_happens = false;
			}
		}
		if (camera[1] + resolution[1] * (camera[2] + delta[2]) >= gm.map_dimensions[1]) {
			if (resolution[0] * (camera[2] + delta[2]) < gm.map_dimensions[0] && resolution[1] * (camera[2] + delta[2]) < gm.map_dimensions[1]) {
				camera[1] = gm.map_dimensions[1] - (unsigned int)ceilf(resolution[1] * (camera[2] + delta[2])) - 1;
			} else {
				delta_z_happens = false;
			}
		}
		if (delta_z_happens) camera[2] += delta[2];
	}

	if (camera[0] + delta[0] + resolution[0] * camera[2] >= gm.map_dimensions[0]) {
		camera[0] = gm.map_dimensions[0] - (unsigned int)ceilf(resolution[0] * camera[2]) - 1;
	} else {
		camera[0] += delta[0];
	}
	if (camera[0] + delta[0] < 0) {
		camera[0] = 0;
	}
	if (camera[1] + delta[1] + resolution[1] * camera[2] >= gm.map_dimensions[1]) {
		camera[1] = gm.map_dimensions[1] - (unsigned int)ceil(resolution[1] * camera[2]) - 1;
	} else {
		camera[1] += delta[1];
	}
	if (camera[1] + delta[1] < 0) {
		camera[1] = 0;
	}
}

void camera_get_crop(vector<unsigned int> &out_crop) {
	out_crop[0] = camera[0];
	out_crop[1] = camera[0] + (unsigned int)floorf(resolution[0] * camera[2]);
	out_crop[2] = camera[1];
	out_crop[3] = camera[1] + (unsigned int)floorf(resolution[1] * camera[2]);
}