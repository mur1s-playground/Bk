#include "Player.hpp"

#include <vector>
#include <random>
#include <ctime>

map<string, struct player> players;

void player_add(string name, enum player_type pt, unsigned int model_id) {
    map<string, struct player>::iterator it = players.find(name);
    if (it != players.end()) {
    } else {
        struct player p;
        p.pt = pt;
        for (int i = 0; i < name.length() && i < 50; i++) {
            p.name[i] = name[i];
            p.name_len = i;
        }
        p.health = 100;
        for (int i = 0; i < 6; i++) {
            p.inventory[i].item_id = UINT_MAX;
            p.inventory[i].item_param = 0;
        }
        p.player_stance = PS_WALKING;
        p.player_action = PA_NONE;
        p.orientation = (float)(rand() % 360);
        p.model_id = model_id;
        players.try_emplace(name, p);
    }
}

void player_type_change(string name, enum player_type pt) {
    map<string, struct player>::iterator it = players.find(name);
    if (it != players.end()) {
        it->second.pt = pt;
    }
}