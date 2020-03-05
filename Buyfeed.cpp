#include "Buyfeed.hpp"

#include "Player.hpp"

#include <math.h>

int buy_feed_count = 0;
int buy_feed_pos = 0;
struct buy_feed_element* buy_feed;

map<string, int>	buy_feed_map;

void buyfeed_init(struct bit_field* bf_rw) {
	int size = players.size() * sizeof(struct buy_feed_element);
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));

	buy_feed_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;

	ui_value_as_config(bf_rw, "ingame_overlay", "buyfeed", 0, buy_feed_pos);
}

void buyfeed_add(struct bit_field* bf_rw, char buyer[50], int item_id) {
	struct buy_feed_element* bf = (struct buy_feed_element*) &bf_rw->data[buy_feed_pos];

	map<string, int>::iterator bfm_it = buy_feed_map.find(string(buyer));
	if (bfm_it != buy_feed_map.end()) {
		
		for (int i = 0; i < 63; i++) {
			if (bf[bfm_it->second].value[i] == '\0') {
				if (item_id == 51) {
					bf[bfm_it->second].value[i] = ']';
				}
				else if (item_id == 52) {
					bf[bfm_it->second].value[i] = '`';
				}
				bf[bfm_it->second].value[i + 1] = '^';
				break;
			}
		}
		
		if (bfm_it->second < buy_feed_count - 1) {
			char bfm_tmp[64];
			memcpy(&bfm_tmp, &bf[bfm_it->second], sizeof(struct buy_feed_element));

			for (int i = bfm_it->second; i < buy_feed_count - 1; i++) {
				memcpy(&bf[i], &bf[i + 1], sizeof(struct buy_feed_element));
			}
		
			memcpy(&bf[buy_feed_count - 1], &bfm_tmp, sizeof(struct buy_feed_element));
			
			map<string, int>::iterator bfm_it2 = buy_feed_map.begin();
			while (bfm_it2 != buy_feed_map.end()) {
				if (bfm_it2->second > bfm_it->second) {
					bfm_it2->second--;
				}
				bfm_it2++;
			}
			bfm_it->second = buy_feed_count - 1;
		}
	} else {
		struct buy_feed_element bfe(buyer, item_id);
		memcpy(&bf[buy_feed_count], &bfe, sizeof(struct buy_feed_element));

		buy_feed_map.emplace(string(buyer), buy_feed_count);
		buy_feed_count++;
	}

	int size = players.size() * sizeof(struct buy_feed_element);
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));

	bit_field_invalidate_bulk(bf_rw, buy_feed_pos, size_in_bf);

	ui_value_as_config(bf_rw, "ingame_overlay", "buyfeed", 1, buy_feed_count);
}

void buyfeed_reset(struct bit_field* bf_rw) {
	bit_field_remove_bulk_from_segment(bf_rw, buy_feed_pos - 1);
	buy_feed_count = 0;
	buy_feed_pos = 0;
}