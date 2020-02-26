#include "KillFeed.hpp"

#include "Player.hpp"

int kill_count = 0;
int kill_feed_pos = 0;
struct kill_feed_element* kill_feed;

void killfeed_init(struct bit_field* bf_rw) {
	int size = players.size() * sizeof(struct kill_feed_element);
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));

	kill_feed_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf)+1;
}

void killfeed_add(struct bit_field* bf_rw, char shooter[50], char victim[50], bool storm) {
	struct kill_feed_element kfe(shooter, victim, storm);
	struct kill_feed_element* kf = (struct kill_feed_element*) &bf_rw->data[kill_feed_pos];

	memcpy(&kf[kill_count], &kfe, sizeof(struct kill_feed_element));

	int size = players.size() * sizeof(struct kill_feed_element);
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));

	bit_field_invalidate_bulk(bf_rw, kill_feed_pos, size_in_bf);

	kill_count++;

	ui_value_as_config(bf_rw, "ingame_overlay", "killfeed", 1, kill_count);
}