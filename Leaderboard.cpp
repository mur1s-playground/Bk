#include "Leaderboard.hpp"

#include "Player.hpp"
#include <sstream>

unsigned int leaderboard_count = 0;

unsigned int leaderboard_place_pos = 0;
unsigned int leaderboard_player_pos = 0;
unsigned int leaderboard_dmg_pos = 0;
unsigned int leaderboard_kills_pos = 0;
unsigned int leaderboard_killedby_pos = 0;

unsigned int player_size_in_bf;
unsigned int place_size_in_bf;

void leaderboard_init(struct bit_field* bf_rw) {
	int size = players.size() * sizeof(struct leaderboard_place_element);
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
	place_size_in_bf = size_in_bf;
	leaderboard_place_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_place", 0, leaderboard_place_pos);
	
	size = players.size() * sizeof(struct playerlist_element);
	size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
	player_size_in_bf = size_in_bf;
	leaderboard_player_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_player", 0, leaderboard_player_pos);
	
	size = players.size() * sizeof(struct leaderboard_place_element);
	size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
	leaderboard_dmg_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_dmg", 0, leaderboard_dmg_pos);
	
	size = players.size() * sizeof(struct leaderboard_place_element);
	size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
	leaderboard_kills_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_kills", 0, leaderboard_kills_pos);

	size = players.size() * sizeof(struct playerlist_element);
	size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
	leaderboard_killedby_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_killedby", 0, leaderboard_killedby_pos);
	
}

void leaderboard_add(struct bit_field* bf_rw, const char playername[50], const unsigned int dmg, const unsigned int kills, const char shooter[50]) {
	stringstream ss_place;
	ss_place << (players.size() - leaderboard_count);
	struct leaderboard_place_element lpe(ss_place.str().c_str());
	struct leaderboard_place_element* lp = (struct leaderboard_place_element*) &bf_rw->data[leaderboard_place_pos];
	memcpy(&lp[leaderboard_count], &lpe, sizeof(struct leaderboard_place_element));
	bit_field_invalidate_bulk(bf_rw, leaderboard_place_pos, place_size_in_bf);
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_place", 1, leaderboard_count+1);
	
	struct playerlist_element ple(playername);
	struct playerlist_element* pl = (struct playerlist_element*) &bf_rw->data[leaderboard_player_pos];
	memcpy(&pl[leaderboard_count], &ple, sizeof(struct playerlist_element));
	bit_field_invalidate_bulk(bf_rw, leaderboard_player_pos, player_size_in_bf);
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_player", 1, leaderboard_count + 1);
	
	stringstream ss_dmg;
	ss_dmg << dmg;
	struct leaderboard_place_element lde(ss_dmg.str().c_str());
	struct leaderboard_place_element* ld = (struct leaderboard_place_element*) &bf_rw->data[leaderboard_dmg_pos];
	memcpy(&ld[leaderboard_count], &lde, sizeof(struct leaderboard_place_element));
	bit_field_invalidate_bulk(bf_rw, leaderboard_dmg_pos, place_size_in_bf);
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_dmg", 1, leaderboard_count + 1);
	
	stringstream ss_kills;
	ss_kills << kills;
	struct leaderboard_place_element lke(ss_kills.str().c_str());
	struct leaderboard_place_element* lk = (struct leaderboard_place_element*) &bf_rw->data[leaderboard_kills_pos];
	memcpy(&lk[leaderboard_count], &lke, sizeof(struct leaderboard_place_element));
	bit_field_invalidate_bulk(bf_rw, leaderboard_kills_pos, place_size_in_bf);
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_kills", 1, leaderboard_count + 1);
	
	struct playerlist_element pkbe(shooter);
	struct playerlist_element* pkb = (struct playerlist_element*) &bf_rw->data[leaderboard_killedby_pos];
	memcpy(&pkb[leaderboard_count], &pkbe, sizeof(struct playerlist_element));
	bit_field_invalidate_bulk(bf_rw, leaderboard_killedby_pos, player_size_in_bf);
	ui_value_as_config(bf_rw, "ingame_menu", "leaderboard_killedby", 1, leaderboard_count + 1);
	
	leaderboard_count++;
}

void leaderboard_reset(struct bit_field *bf_rw) {
	bit_field_remove_bulk_from_segment(bf_rw, leaderboard_place_pos - 1);
	bit_field_remove_bulk_from_segment(bf_rw, leaderboard_player_pos - 1);
	bit_field_remove_bulk_from_segment(bf_rw, leaderboard_dmg_pos - 1);
	bit_field_remove_bulk_from_segment(bf_rw, leaderboard_kills_pos - 1);
	bit_field_remove_bulk_from_segment(bf_rw, leaderboard_killedby_pos - 1);
	leaderboard_place_pos = 0;
	leaderboard_player_pos = 0;
	leaderboard_dmg_pos = 0;
	leaderboard_kills_pos = 0;
	leaderboard_killedby_pos = 0;
	leaderboard_count = 0;
}