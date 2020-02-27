#ifndef PLAYER_HPP
#define PLAYER_HPP

#include <map>
#include <string>
#include "Vector2.hpp"
#include "Vector3.hpp"
#include "BitField.hpp"
#include "Item.hpp"

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

struct player {
	enum player_type pt;

	char name[50];
	unsigned int name_len;

	bool					alive;

	int						damage_dealt;
	int						kills;

	int						health;
	int						shield;

	unsigned int			inventory_active_slot;
	struct item				inventory[6];

	enum player_stance		player_stance;
	enum player_action		player_action;

	unsigned int			entity_id;
};

extern unsigned int						player_models_position;
extern map<player_type, struct model>	player_models;

extern map<string, struct player>		players;
extern unsigned int						players_position;

extern unsigned int						players_max;

void player_models_init(struct bit_field* bf_assets);
void player_add(string name, enum player_type pt, unsigned int entity_id);
void player_type_change(string name, enum player_type pt);
void players_upload(struct bit_field* bf);

#endif