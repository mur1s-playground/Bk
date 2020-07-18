#ifndef PLAYER_HPP
#define PLAYER_HPP

#include "FeatureToggles.hpp"
#include <map>
#include <atomic>
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
	PAT_NONE,
	PAT_BUY_ITEM,
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

	struct player *			attack_target;

	enum player_action_type move_reason;
	vector2<float>			move_target;

	unsigned int			move_path_len;
	unsigned int			move_path_active_id;
	vector2<float>			move_path[10];
#ifndef BRUTE_PATHING
	unsigned int			pathing_position;
	int						pathing_calc_stage;
#endif
	unsigned int			actions;
	unsigned int			action_params[50];

	unsigned int			entity_id;
};

extern unsigned int						player_models_position;
extern map<player_type, struct model>	player_models;

extern map<string, struct player>		players;
extern unsigned int						players_position;

extern unsigned int						players_max;

extern unsigned int        				player_selected_id;

extern atomic<bool>						player_move_target_override_set;
extern vector2<float>					player_move_target_override;

void player_models_init(struct bit_field* bf_assets);
void player_add(struct bit_field* bf_rw, string name, enum player_type pt, unsigned int entity_id);
void player_type_change(string name, enum player_type pt);
void player_action_param_add(struct player* pl, const enum player_action_type pat, const unsigned int param1, const unsigned int param2);
void players_upload(struct bit_field* bf);
bool players_process_left_click(vector2<unsigned int> position);
void players_process_right_click(vector2<unsigned int> position);

#endif