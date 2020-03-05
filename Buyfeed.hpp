#ifndef BUYFEED_HPP
#define BUYFEED_HPP

#include "BitField.hpp"
#include "UI.hpp"

#include <map>

struct buy_feed_element {
    buy_feed_element(char s[50], int item_id) {
        int cur_pos = 0;
        value[cur_pos++] = '^';
        for (int i = 0; i < 50; i++) {
            if (s[i] == '\0') break;
            value[cur_pos] = s[i];
            cur_pos++;
        }
        value[cur_pos++] = '^';
        if (item_id == 51) {
            value[cur_pos++] = ']';
        } else if (item_id == 52) {
            value[cur_pos++] = '`';
        }
        value[cur_pos++] = '^';
        for (int i = cur_pos; i < 64; i++) {
            value[i] = '\0';
        }
    }
    char value[64];
};

using namespace std;

extern int buy_feed_count;
extern int buy_feed_pos;
extern struct buy_feed_element* buy_feed;

void buyfeed_init(struct bit_field* bf_rw);
void buyfeed_add(struct bit_field* bf_rw, char buyer[50], int item_id);
void buyfeed_reset(struct bit_field* bf_rw);

#endif