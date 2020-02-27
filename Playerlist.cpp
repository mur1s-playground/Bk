#include "Playerlist.hpp"

#include "Player.hpp"

int playerlist_count;
int playerlist_pos;
struct playerlist_element* playerlist;

void playerlist_init(struct bit_field* bf_rw) {
	int size = players.size() * sizeof(struct playerlist_element);
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));

	playerlist_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;
}

void playerlist_add(struct bit_field* bf_rw, const char playername[50]) {
	struct playerlist_element pes(playername);
	struct playerlist_element* pe = (struct playerlist_element*) & bf_rw->data[playerlist_pos];

	memcpy(&pe[playerlist_count], &pes, sizeof(struct playerlist_element));

	int size = players.size() * sizeof(struct playerlist_element);
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));

	bit_field_invalidate_bulk(bf_rw, playerlist_pos, size_in_bf);

	playerlist_count++;

	ui_value_as_config(bf_rw, "lobby", "players", 1, playerlist_count);
}