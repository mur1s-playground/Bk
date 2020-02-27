#ifndef PLAYERLIST_HPP
#define PLAYERLIST_HPP

#include <vector>
#include <utility>
#include "UI.hpp"

struct playerlist_element {
    playerlist_element(const char s[50]) {
        int cur_pos = 0;
        value[cur_pos++] = '^';
        for (int i = 0; i < 50; i++) {
            if (s[i] == '\0') break;
            value[cur_pos] = s[i];
            cur_pos++;
        }
        value[cur_pos++] = '^';
        for (int i = cur_pos; i < 52; i++) {
            value[i] = '\0';
        }
    }
    char value[52];
};

using namespace std;

extern int playerlist_count;
extern int playerlist_pos;
extern struct playerlist_element* playerlist;

void playerlist_init(struct bit_field* bf_rw);
void playerlist_add(struct bit_field* bf_rw, const char playername[50]);

#endif