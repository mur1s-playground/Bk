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

extern map<string, struct player> players;

void player_add(string name, enum player_type pt, unsigned int model_id);
void player_type_change(string name, enum player_type pt);

#endif