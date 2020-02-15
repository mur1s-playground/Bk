#include "Camera.hpp"

#include "Main.hpp"

void camera_move(struct vector3<float> delta) {
	if (camera[0] + delta[0] + (unsigned int)floorf(resolution[0] * camera[2]) < map_dimensions[0] && camera[0] + delta[0] >= 0) camera[0] += delta[0];
	if (camera[1] + delta[1] + (unsigned int)floorf(resolution[1] * camera[2]) < map_dimensions[1] && camera[1] + delta[1] >= 0) camera[1] += delta[1];
	if (camera[0] + (unsigned int)floorf(resolution[0] * (camera[2] + delta[3])) < map_dimensions[0] &&
		camera[1] + (unsigned int)floorf(resolution[1] * (camera[2] + delta[3])) < map_dimensions[1] && camera[2] + delta[2] >= 0.01f) camera[2] += delta[2];
}

void camera_get_crop(vector<unsigned int> &out_crop) {
	out_crop[0] = camera[0];
	out_crop[1] = camera[0] + (unsigned int)floorf(resolution[0] * camera[2]);
	out_crop[2] = camera[1];
	out_crop[3] = camera[1] + (unsigned int)floorf(resolution[1] * camera[2]);
}