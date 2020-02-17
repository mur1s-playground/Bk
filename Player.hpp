#ifndef PLAYER_HPP
#define PLAYER_HPP

#include <map>
#include <string>
#include "Vector2.hpp"
#include "Vector3.hpp"

using namespace std;

enum player_type {
	PT_HOPE,
};

enum player_stance {
	PS_STANDING,
	PS_WALKING
};

enum player_action {
	PA_NONE,
	PA_SHOOTING
};

struct player_item {
	unsigned int	item_id;
	int				item_param;
};

struct player {
	enum player_type pt;

	char name[50];
	unsigned int name_len;

	struct vector3<float> position;
	float orientation;
	int health;

	struct player_item inventory[6];

	enum player_stance player_stance;
	enum player_action player_action;

	unsigned int model_id;
};

struct player_model {
	unsigned int id;

	float model_scale;
	unsigned int model_positions;

	struct vector3<unsigned int> model_dimensions;
	unsigned int model_med_positions;
	unsigned int model_lo_positions;


	float shadow_scale;
	struct vector3<unsigned int> shadow_dimensions;
	struct vector2<unsigned int> shadow_offset;
	unsigned int shadow_positions;
	unsigned int shadow_med_positions;
	unsigned int shadow_lo_positions;
};

extern map<string, struct player> players;

void player_add(string name, enum player_type pt);
void player_type_change(string name, enum player_type pt);

#endif