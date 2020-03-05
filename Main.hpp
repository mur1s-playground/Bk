#ifndef MAIN_HPP
#define MAIN_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"
#include <vector>

#include "Grid.hpp"
#include "Model.hpp"
#include "Player.hpp"

extern struct grid gd;

extern struct bit_field bf_assets;
extern struct bit_field bf_rw;
extern struct bit_field bf_output;

extern struct vector3<float> camera;
extern struct vector2<unsigned int> resolution;

extern int target_ticks_per_second;

extern bool game_started;
extern bool irc_started;
extern bool running;

extern map<string, int>				bits_spent;
extern map<string, int>				bits_shield;
extern map<string, int>				bits_bandage;

#endif