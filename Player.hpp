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

enum player_action_type {
	PAT_PICKUP_ITEM,
	PAT_SHOOT_AT,
	PAT_MOVE
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

	unsigned int			actions;
	unsigned int			action_params[50];

	unsigned int			entity_id;
};

extern unsigned int						player_models_position;
extern map<player_type, struct model>	player_models;

extern map<string, struct player>		players;
extern unsigned int						players_position;

extern unsigned int						players_max;

void player_models_init(struct bit_field* bf_assets);
void player_add(struct bit_field* bf_rw, string name, enum player_type pt, unsigned int entity_id);
void player_type_change(string name, enum player_type pt);
void player_action_param_add(struct player* pl, const enum player_action_type pat, const unsigned int param1, const unsigned int param2);
void players_upload(struct bit_field* bf);


#endif