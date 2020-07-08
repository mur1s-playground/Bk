#include "Player.hpp"

#include <vector>
#include <random>
#include <ctime>
#include <sstream>
#include "Util.hpp"
#include "Model.hpp"
#include "AssetLoader.hpp"
#include "Entity.hpp"
#include "Playerlist.hpp"
#include "Main.hpp"

unsigned int                    player_models_position;
map<player_type, struct model>  player_models;

unsigned int                    players_position;
map<string, struct player>      players;

unsigned int				    players_max = 0;

unsigned int        			player_selected_id = -1;

atomic<bool>                    player_move_target_override_set;
vector2<float>                  player_move_target_override = { 0.0f, 0.0f };

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

void player_add(struct bit_field *bf_rw, string name, enum player_type pt, unsigned int entity_id) {
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
        p.move_reason = PAT_NONE;
        p.move_target = { -1.0f , -1.0f };
        p.move_path_len = 10;
        p.move_path_active_id = 10;
        p.actions = 0;
        p.entity_id = entity_id;
        players.try_emplace(name, p);
        playerlist_add(bf_rw, name.c_str());
    }
}

void player_type_change(string name, enum player_type pt) {
    map<string, struct player>::iterator it = players.find(name);
    if (it != players.end()) {
        it->second.pt = pt;
    }
}

void player_action_param_add(struct player* pl, const enum player_action_type pat, const unsigned int param1, const unsigned int param2) {
    unsigned int* pap_start = &pl->action_params[3 * pl->actions];
    if (pl->actions < 50 / 3) {
        pap_start[0] = pat;
        pap_start[1] = param1;
        pap_start[2] = param2;
        pl->actions++;
    }
}

void players_upload(struct bit_field *bf) {
    player_move_target_override_set.store(false);
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

bool players_process_left_click(vector2<unsigned int> position) {
    WaitForSingleObject(bf_rw.device_locks[0], INFINITE);
    //printf("position: %d %d\n", position[0], position[1]);
    map<string, struct player>::iterator it = players.begin();
    int closest_e_id = -1;
    float closest_dist = 10000000.0f;
    struct player* pl_closest;
    while (it != players.end()) {
        struct player* pl = &it->second;
        if (pl->alive) {
            if (pl->entity_id < UINT_MAX) {
                struct entity* es = (struct entity*)&bf_rw.data[entities_position];
                struct entity* en = (struct entity*)&es[pl->entity_id];

                //printf("en pos: %f %f\n", en->position[0], en->position[1]);

                float dist = sqrtf((en->position[0]+8 - (float)position[0])* (en->position[0]+8 - (float)position[0]) + (en->position[1]+8 - (float)position[1])* (en->position[1]+8 - (float)position[1]));
                if (dist < closest_dist) {
                    pl_closest = pl;
                    closest_dist = dist;
                    closest_e_id = pl->entity_id;
                }
           }
        }
        it++;
    }
    //printf("closest dist %f\n", closest_dist);
    if (closest_dist < 16 && closest_e_id > -1) {
        player_selected_id = closest_e_id;
        printf("move_target: %f %f\n", pl_closest->move_target[0], pl_closest->move_target[1]);
        printf("move_reason: %d\n", pl_closest->move_reason);
        printf("move_path_len: %d\n", pl_closest->move_path_len);
        printf("move_path_active: %d\n", pl_closest->move_path_active_id);
    } else {
        player_selected_id = -1;
    }
    ReleaseMutex(bf_rw.device_locks[0]);
    if (closest_dist < 16 && closest_e_id > -1) {
        return true;
    }
    return false;
}

void players_process_right_click(vector2<unsigned int> position) {
    player_move_target_override = {(float)position[0], (float)position[1]};
    player_move_target_override_set.store(true);
}