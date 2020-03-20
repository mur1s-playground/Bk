#ifndef LEADERBOARD_HPP
#define LEADERBOARD_HPP

#include <vector>
#include <utility>
#include "UI.hpp"
#include "BitField.hpp"

#include "Playerlist.hpp"

struct leaderboard_place_element {
    leaderboard_place_element(const char s[8]) {
        int cur_pos = 0;
        value[cur_pos++] = 32;
        for (int i = 0; i < 8; i++) {
            if (s[i] == '\0') break;
            value[cur_pos] = s[i];
            cur_pos++;
        }
        value[cur_pos++] = 32;
        for (int i = cur_pos; i < 10; i++) {
            value[i] = '\0';
        }
    }
    char value[10];
};

using namespace std;

void leaderboard_init(struct bit_field* bf_rw);
void leaderboard_add(struct bit_field* bf_rw, const char playername[50], const unsigned int dmg, const unsigned int kills, const char shooter[50]);
void leaderboard_reset(struct bit_field* bf_rw);

#endif