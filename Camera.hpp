#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "Vector3.hpp"
#include <vector>

using namespace std;

extern struct vector3<float> camera;

extern vector<unsigned int> camera_crop;

void camera_init();

void camera_move_z_tick();
void camera_move(struct vector3<float> delta);
void camera_get_crop(vector<unsigned int>& out_crop);

#endif