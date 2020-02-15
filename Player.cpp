#include "Player.hpp"

#include <vector>

map<string, struct player> players;

void player_add(string name, enum player_type pt) {
    map<string, struct player>::iterator it = players.find(name);
    if (it != players.end()) {
    } else {
        struct player p;
        p.pt = pt;
        for (int i = 0; i < name.length() && i < 50; i++) {
            p.name[i] = name[i];
            p.name_len = i;
        }
        players.try_emplace(name, p);
    }
}

void player_type_change(string name, enum player_type pt) {
    map<string, struct player>::iterator it = players.find(name);
    if (it != players.end()) {
        it->second.pt = pt;
    }
}