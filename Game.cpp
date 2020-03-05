#include "Game.hpp"

#include <sstream>
#include <math.h>

#include "Main.hpp"
#include "Grid.hpp"
#include "Entity.hpp"
#include "KillFeed.hpp"
#include "Buyfeed.hpp"
#include "Leaderboard.hpp"
#include "Map.hpp"
#include "Storm.hpp"
#include "UI.hpp"
#include "AssetLoader.hpp"
#include "TwitchIntegration.hpp"

void game_init() {
	printf("initialising game\n");

	//load first map
	map_load(&bf_assets, ui_textfield_get_value(&bf_rw, "lobby", "selected_map"));

	//uploading assets to gpu
	bit_field_update_device(&bf_assets, 0);

	//entity grid
	grid_init(&bf_rw, &gd, struct vector3<float>((float)gm.map_dimensions[0], (float)gm.map_dimensions[1], 1.0f), struct vector3<float>(32.0f, 32.0f, 1.0f), struct vector3<float>(0, 0, 0));

	map_add_static_assets(&bf_assets, &bf_rw, &gd);

	storm_init(&bf_assets, &bf_rw);

	printf("initialization finished\n");
}

void game_start() {
	printf("starting game\n");

	camera = { 0.0f, 0.0f, 1.0f };

	//spawn players
	map<string, struct player>::iterator pl_it = players.begin();
	int i = 0;
	while (pl_it != players.end()) {
		//for (int i = 0; i < players.size(); i++) {
		stringstream ss_p;
		ss_p << i;

		entity_add(string(pl_it->second.name), ET_PLAYER, 0, 0);
		struct entity* cur_e = &entities[entities.size() - 1];

		cur_e->params[0] = (char)pl_it->second.health;
		cur_e->params[1] = (char)pl_it->second.shield;
		int* params = (int*)&cur_e->params;
		int params_pos = 1;
		for (int ip = 0; ip < 6; ip++) {
			params[params_pos++] = pl_it->second.inventory->item_id;
			params[params_pos++] = pl_it->second.inventory->item_param;
		}

		bool found_spawn = false;
		float x = 0;
		float y = 0;
		float z = 0;
		while (!found_spawn) {
			x = 10.0f + rand() % (gm.map_dimensions[0] - 32);
			y = 10.0f + rand() % (gm.map_dimensions[1] - 32);
			z = 0.0f;
			unsigned int grid_index = grid_get_index(bf_rw.data, gd.position_in_bf, { x, y, z });
			if (bf_rw.data[gd.data_position_in_bf + 1 + grid_index] == 0) {
				unsigned char* pathables = (unsigned char*)&bf_assets.data[gm.map_pathable_position];
				unsigned char pathable = pathables[(int)floorf(y) * gm.map_dimensions[0] + (int)floorf(x)];
				if (pathable > 0) {
					unsigned char* spawn_probabilities = (unsigned char*)&bf_assets.data[gm.map_spawn_probabilities_position];
					unsigned char spawn_probability = spawn_probabilities[(int)floorf(y) * gm.map_dimensions[0] + (int)floorf(x)];
					if (rand() / (float)RAND_MAX * 255 <= spawn_probability) {
						found_spawn = true;
					}
				}
			}
		}
		cur_e->position = { x, y, z };
		struct vector3<float> max_pos = model_get_max_position(&player_models[PT_HOPE]);
		//object itself
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, cur_e->position, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, max_pos, entities.size() - 1);
		//object text
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { cur_e->position[0] - 32.0f - 3, cur_e->position[1] - 32.0f - 3, cur_e->position[2] - 0.0f }, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { cur_e->name_len * 32.0f + 32.0f + 3, 96.0f + 3, 0 }, entities.size() - 1);
		//object inventory
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { cur_e->position[0] - 32.0f - 3, cur_e->position[1], cur_e->position[2] - 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 32.0f, 32.0f * 6, 0 }, entities.size() - 1);
		pl_it->second.entity_id = entities.size() - 1;

		i++;
		pl_it++;
	}

	bit_field_update_device(&bf_assets, 0);
	//players_upload(&bf_rw);

	killfeed_init(&bf_rw);

	leaderboard_init(&bf_rw);

	buyfeed_init(&bf_rw);

	//spawn weapons
	unsigned int weapon_count = max(100, (int)players.size());
	for (int i = 0; i < weapon_count; i++) {
		stringstream ss_p;
		ss_p << i;
		entity_add("colt_" + ss_p.str(), ET_ITEM, 50, 0);
		struct entity* cur_e = &entities[entities.size() - 1];
		bool found_spawn = false;
		float x = 0;
		float y = 0;
		float z = 0;
		while (!found_spawn) {
			x = 10.0f + rand() % (gm.map_dimensions[0] - 32);
			y = 10.0f + rand() % (gm.map_dimensions[1] - 32);
			z = 0.0f;
			unsigned int grid_index = grid_get_index(bf_rw.data, gd.position_in_bf, { x, y, z });
			if (bf_rw.data[gd.data_position_in_bf + 1 + grid_index] == 0) {
				unsigned char* pathables = (unsigned char*)&bf_assets.data[gm.map_pathable_position];
				unsigned char pathable = pathables[(int)floorf(y) * gm.map_dimensions[0] + (int)floorf(x)];
				if (pathable > 0) {
					unsigned char* loot_spawn_probabilities = (unsigned char*)&bf_assets.data[gm.map_loot_probabilities_position];
					unsigned char loot_spawn_probability = loot_spawn_probabilities[(int)floorf(y) * gm.map_dimensions[0] + (int)floorf(x)];
					if (rand() / (float)RAND_MAX * 255 <= loot_spawn_probability) {
						found_spawn = true;
					}
				}
			}
		}
		cur_e->position = { x, y, z };
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, cur_e->position, { item_models[0].model_scale, item_models[0].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&item_models[0]), entities.size() - 1);
	}

	//spawn shield
	unsigned int shield_count = max(100, (int)players.size());
	for (int i = 0; i < shield_count; i++) {
		stringstream ss_p;
		ss_p << i;
		entity_add("bottle_" + ss_p.str(), ET_ITEM, 51, 0);
		struct entity* cur_e = &entities[entities.size() - 1];
		bool found_spawn = false;
		float x = 0;
		float y = 0;
		float z = 0;
		while (!found_spawn) {
			x = 10.0f + rand() % (gm.map_dimensions[0] - 32);
			y = 10.0f + rand() % (gm.map_dimensions[1] - 32);
			z = 0.0f;
			unsigned int grid_index = grid_get_index(bf_rw.data, gd.position_in_bf, { x, y, z });
			if (bf_rw.data[gd.data_position_in_bf + 1 + grid_index] == 0) {
				unsigned char* pathables = (unsigned char*)&bf_assets.data[gm.map_pathable_position];
				unsigned char pathable = pathables[(int)floorf(y) * gm.map_dimensions[0] + (int)floorf(x)];
				if (pathable > 0) {
					unsigned char* loot_spawn_probabilities = (unsigned char*)&bf_assets.data[gm.map_loot_probabilities_position];
					unsigned char loot_spawn_probability = loot_spawn_probabilities[(int)floorf(y) * gm.map_dimensions[0] + (int)floorf(x)];
					if (rand() / (float)RAND_MAX * 255 <= loot_spawn_probability) {
						found_spawn = true;
					}
				}
			}
		}
		cur_e->position = { x, y, z };
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, cur_e->position, { item_models[1].model_scale, item_models[1].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&item_models[1]), entities.size() - 1);
	}

	//spawn bandages
	unsigned int bandages_count = max(100, (int)players.size());
	for (int i = 0; i < bandages_count; i++) {
		stringstream ss_p;
		ss_p << i;
		entity_add("bandage_" + ss_p.str(), ET_ITEM, 52, 0);
		struct entity* cur_e = &entities[entities.size() - 1];
		bool found_spawn = false;
		float x = 0;
		float y = 0;
		float z = 0;
		while (!found_spawn) {
			x = 10.0f + rand() % (gm.map_dimensions[0] - 32);
			y = 10.0f + rand() % (gm.map_dimensions[1] - 32);
			z = 0.0f;
			unsigned int grid_index = grid_get_index(bf_rw.data, gd.position_in_bf, { x, y, z });
			if (bf_rw.data[gd.data_position_in_bf + 1 + grid_index] == 0) {
				unsigned char* pathables = (unsigned char*)&bf_assets.data[gm.map_pathable_position];
				unsigned char pathable = pathables[(int)floorf(y) * gm.map_dimensions[0] + (int)floorf(x)];
				if (pathable > 0) {
					unsigned char* loot_spawn_probabilities = (unsigned char*)&bf_assets.data[gm.map_loot_probabilities_position];
					unsigned char loot_spawn_probability = loot_spawn_probabilities[(int)floorf(y) * gm.map_dimensions[0] + (int)floorf(x)];
					if (rand() / (float)RAND_MAX * 255 <= loot_spawn_probability) {
						found_spawn = true;
					}
				}
			}
		}
		cur_e->position = { x, y, z };
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, cur_e->position, { item_models[2].model_scale, item_models[2].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&item_models[2]), entities.size() - 1);
	}

	entities_upload(&bf_rw);

	bit_field_update_device(&bf_rw, 0);
}

void game_destroy() {
	//unload map
	asset_loader_unload_by_containing(&bf_assets, "./maps/" + ui_textfield_get_value(&bf_rw, "lobby", "selected_map") + "/");
	map_models.clear();
	bit_field_remove_bulk_from_segment(&bf_assets, map_models_position-1);

	//unload grid
	bit_field_remove_bulk_from_segment(&bf_rw, gd.data_position_in_bf);
	bit_field_remove_bulk_from_segment(&bf_rw, gd.position_in_bf);

	//reset storm
	storm_destroy();

	//kill irc
	twitch_terminate_irc();
	irc_started = false;

	//remove players
	players.clear();

	//remove playerlist
	bit_field_remove_bulk_from_segment(&bf_rw, playerlist_pos - 1);
	playerlist_count = 0;

	//reset ui elements
	char tmp = '\0';
	top_kills = 0;
	top_damage = 0;
	ui_textfield_set_value(&bf_rw, "ingame_menu", "top_kills", &tmp);
	ui_textfield_set_value(&bf_rw, "ingame_menu", "top_kills_nr", &tmp);
	ui_textfield_set_value(&bf_rw, "ingame_menu", "top_damage", &tmp);
	ui_textfield_set_value(&bf_rw, "ingame_menu", "top_damage_nr", &tmp);
	ui_textfield_set_value(&bf_rw, "ingame_menu", "top_placement", &tmp);
	ui_textfield_set_int(&bf_rw, "lobby", "playercount", 0);

	//remove entities
	entities.clear();
	bit_field_remove_bulk_from_segment(&bf_rw, entities_position - 1);

	//killfeed reset
	killfeed_reset(&bf_rw);

	//leaderboard reset
	leaderboard_reset(&bf_rw);

	//buyfeed reset
	buyfeed_reset(&bf_rw);

	target_ticks_per_second = 30;
}