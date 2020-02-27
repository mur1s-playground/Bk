#include "Player.hpp"

#include <vector>
#include <random>
#include <ctime>
#include <sstream>
#include "Util.hpp"
#include "Model.hpp"
#include "AssetLoader.hpp"
#include "Entity.hpp"

unsigned int                    player_models_position;
map<player_type, struct model>  player_models;

unsigned int                    players_position;
map<string, struct player>      players;

unsigned int				    players_max = 0;

void player_models_init(struct bit_field* bf_assets) {
    vector<string> model_cfgs = get_all_files_names_within_folder("./players", "*", "cfg");
    vector<struct model> pms, pms_sorted;
    for (int i = 0; i < model_cfgs.size(); i++) {
        struct model m = model_from_cfg(bf_assets, "./players/", model_cfgs[i]);
        if (model_cfgs[i].substr(0, model_cfgs[i].find_last_of('.')) == "hope") {
            player_models.try_emplace(PT_HOPE, m);
            pms.push_back(m);
        }
    }
    int counter = 0;
    struct model empty_model;
    empty_model.id = UINT_MAX;
    while (counter < pms.size()) {
        for (int i = 0; i < pms.size(); i++) {
            if (pms[i].id == counter) {
                pms_sorted.push_back(pms[i]);
            }
        }
        if (pms_sorted.size() < counter + 1) {
            pms_sorted.push_back(empty_model);
        }
        counter++;
    }
    unsigned int size = pms_sorted.size() * sizeof(struct model);
    unsigned int size_in_bf = (unsigned int)ceilf(size/(float)sizeof(unsigned int));
    player_models_position = bit_field_add_bulk(bf_assets, (unsigned int*)pms_sorted.data(), size_in_bf, size)+1;
}

void player_add(string name, enum player_type pt, unsigned int entity_id) {
    map<string, struct player>::iterator it = players.find(name);
    if (it != players.end()) {
    } else {
        struct player p;
        p.alive = true;
        p.pt = pt;
        p.damage_dealt = 0;
        p.kills = 0;
        for (int i = 0; i < name.length() && i < 50; i++) {
            p.name[i] = name[i];
            p.name_len = i+1;
        }
        for (int i = name.length(); i < 50; i++) {
            p.name[i] = '\0';
        }
        p.health = 100;
        p.shield = 0;
        for (int i = 0; i < 6; i++) {
            p.inventory[i].item_id = UINT_MAX;
            p.inventory[i].item_param = 0;
        }
        p.player_stance = PS_WALKING;
        p.player_action = PA_NONE;
        p.entity_id = entity_id;
        players.try_emplace(name, p);
    }
}

void player_type_change(string name, enum player_type pt) {
    map<string, struct player>::iterator it = players.find(name);
    if (it != players.end()) {
        it->second.pt = pt;
    }
}

void players_upload(struct bit_field *bf) {
    vector<struct player> pl_v;
    map<string, struct player>::iterator it = players.begin();
    if (it != players.end()) {
        pl_v.push_back(it->second);
        it++;
    }
    unsigned int size = pl_v.size() * sizeof(struct player);
    unsigned int size_in_bf = (unsigned int)ceilf(size / (float)sizeof(unsigned int));
    players_position = bit_field_add_bulk(bf, (unsigned int*)pl_v.data(), size_in_bf, size) + 1;
}