#ifndef MAIN_HPP
#define MAIN_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"
#include <vector>
#include "Model.hpp"
#include "Player.hpp"

extern struct bit_field bf_assets;
extern struct bit_field bf_players;
extern struct bit_field bf_output;

extern struct vector3<float> camera;
extern struct vector2<unsigned int> resolution;

extern int target_ticks_per_second;

#endif