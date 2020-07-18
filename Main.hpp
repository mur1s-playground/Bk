#ifndef MAIN_HPP
#define MAIN_HPP

#include "FeatureToggles.hpp"

#include "Vector2.hpp"
#include "Vector3.hpp"
#include <vector>

#include "Grid.hpp"
#include "Model.hpp"
#include "Player.hpp"

extern struct grid gd;

extern struct bit_field bf_assets;
extern struct bit_field bf_map;
extern struct bit_field bf_rw;
extern struct bit_field bf_output;

#ifndef BRUTE_PATHING
	extern struct bit_field bf_pathing;
#endif

extern struct vector2<unsigned int> resolution;

extern bool map_editor;
extern bool map_editor_update_assets;

extern bool irc_started;
extern bool running;

extern int max_bits_per_game;
extern int bits_per_shield;
extern int bits_per_bandage;
extern map<string, int>				bits_spent;
extern map<string, int>				bits_shield;
extern map<string, int>				bits_bandage;

extern int top_kills;
extern int top_damage;

extern struct vector2<unsigned int> mouse_position;

int myrand();

#endif