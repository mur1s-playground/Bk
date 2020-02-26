#ifndef KILLFEED_HPP
#define KILLFEED_HPP

#include <vector>
#include <utility>
#include "UI.hpp"

struct kill_feed_element {
    kill_feed_element(char s[50], char v[50], bool storm) { 
        int cur_pos = 0;
        if (!storm) {
            for (int i = 0; i < 50; i++) {
                if (s[i] == '\0') break;
                value[cur_pos] = s[i];
                cur_pos++;
            }
            value[cur_pos++] = '^';
            value[cur_pos++] = '=';
            value[cur_pos++] = '^';
        } else {
            value[cur_pos++] = ';';
            value[cur_pos++] = '^';
        }
        for (int i = 0; i < 50; i++) {
            if (v[i] == '\0') break;
            value[cur_pos] = v[i];
            cur_pos++;
        }
        for (int i = cur_pos; i < 103; i++) {
            value[i] = '\0';
        }
    }
    char value[103];
};

using namespace std;

extern int kill_count;
extern int kill_feed_pos;
extern struct kill_feed_element* kill_feed;

void killfeed_init(struct bit_field *bf_rw);
void killfeed_add(struct bit_field* bf_rw, char shooter[50], char victim[50], bool storm = false);

#endif