#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "Vector3.hpp"
#include <vector>

using namespace std;

void camera_move(struct vector3<float> delta);
void camera_get_crop(vector<unsigned int>& out_crop);

#endif