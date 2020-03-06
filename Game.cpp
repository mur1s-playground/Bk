#include "Game.hpp"

#include <sstream>
#include <math.h>
#include <ctime>
#include <thread>
#include <chrono>

#include "Main.hpp"
#include "Camera.hpp"
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


atomic<unsigned int>	game_ticks			= 0;
atomic<unsigned int>	game_ticks_target	= 30;
atomic<bool>			game_setup			= false;
atomic<bool>			game_started		= false;
atomic<bool>			game_tick_ready		= false;

bool					game_cleanup		= false;

DWORD WINAPI game_thread(LPVOID param) {
	unsigned int frame_balancing = 0;
	long tf_l = clock();
	while (running) {

		if (game_setup) {
			game_init();
			game_start();
			camera_get_crop(camera_crop);
			ui_set_active("ingame_overlay");
			game_ticks = 0;
			game_started = true;
			game_setup = false;
		} else if (game_started) {
			WaitForSingleObject(bf_rw.device_locks[0], INFINITE);
			game_tick();
			ReleaseMutex(bf_rw.device_locks[0]);

			game_ticks++;
		
			long tf_3 = clock();
			double frame_time = max((tf_3 - tf_l) / (double)CLOCKS_PER_SEC, 0.001);
			double frame_time_target = 1000.0f / (float)game_ticks_target / 1000.0f;
			if (frame_time < frame_time_target) {
				frame_balancing = (int)roundf((frame_time_target - frame_time) * 1000.0f);
			} else {
				frame_balancing = 0;
			}	
			if (frame_balancing > 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(frame_balancing));
			}
			tf_l = clock();
			if (ui_active == "unloading_game") {
				WaitForSingleObject(bf_rw.device_locks[0], INFINITE);
				game_cleanup = true;
				game_started = false;
				ReleaseMutex(bf_rw.device_locks[0]);
			}
		} else if (game_cleanup) {
			game_cleanup = false;
			WaitForSingleObject(bf_assets.device_locks[0], INFINITE);
			WaitForSingleObject(bf_rw.device_locks[0], INFINITE);
			game_destroy();
			ReleaseMutex(bf_rw.device_locks[0]);
			ReleaseMutex(bf_assets.device_locks[0]);
			ui_set_active("main_menu");
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}
	}
	return NULL;
}

void game_init() {
	printf("initialising game\n");

	//load first map
	map_load(&bf_assets, ui_textfield_get_value(&bf_rw, "lobby", "selected_map"));

	//entity grid
	grid_init(&bf_rw, &gd, struct vector3<float>((float)gm.map_dimensions[0], (float)gm.map_dimensions[1], 1.0f), struct vector3<float>(32.0f, 32.0f, 1.0f), struct vector3<float>(0, 0, 0));

	map_add_static_assets(&bf_assets, &bf_rw, &gd);

	storm_init(&bf_assets, &bf_rw);

	printf("initialization finished\n");
}

void game_start() {
	printf("starting game\n");

	camera = { 0.0f, 0.0f, 1.0f };

	printf("spawning players\n");
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

	printf("initialising feeds\n");
	killfeed_init(&bf_rw);

	leaderboard_init(&bf_rw);

	buyfeed_init(&bf_rw);

	printf("spawning items\n");
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

	printf("uploading entities\n");
	entities_upload(&bf_rw);

	printf("updating assets players\n");
	bit_field_update_device(&bf_assets, 0);

	printf("game start successful\n");
}

void game_tick() {
		map<string, struct player>::iterator players_it = players.begin();
		float player_dist_per_tick = 1 / 5.0f;
		int orientation_change_per_tick = 3;
		while (players_it != players.end()) {
			//struct entity* es = (struct entity*) & bf_rw.data[entities_position];
			struct player* pl = &players_it->second;
			if (pl->alive) {
				if (pl->entity_id < UINT_MAX) {
					struct entity* es = (struct entity*) & bf_rw.data[entities_position];
					struct entity* en = (struct entity*) & es[pl->entity_id];

					//struct entity* en = &es[pl->entity_id];
					int has_inv_space = 0;
					int has_gun = -1;
					for (int inv = 0; inv < 6; inv++) {
						if (pl->inventory[inv].item_id == UINT_MAX) {
							has_inv_space++;
						}
						else if (pl->inventory[inv].item_id == 50) {
							has_gun = inv;
							if (pl->inventory[inv].item_param % 15 != 0) {
								pl->inventory[inv].item_param++;
							}
						}
						else if (pl->inventory[inv].item_id == 51) {
							if (pl->shield <= 75 && pl->inventory[inv].item_param != 0) {
								//printf("shiedling\n");
								pl->shield += 25;
								pl->inventory[inv].item_param--;
							}
							if (pl->inventory[inv].item_param == 0) {
								pl->inventory[inv].item_id = UINT_MAX;
								has_inv_space++;
							}
						}
						else if (pl->inventory[inv].item_id == 52) {
							if (pl->health <= 75 && pl->inventory[inv].item_param != 0) {
								//printf("healing\n");
								pl->health += 25;
								pl->inventory[inv].item_param--;
							}
							if (pl->inventory[inv].item_param == 0) {
								pl->inventory[inv].item_id = UINT_MAX;
								has_inv_space++;
							}
						}
					}

					if ((has_gun >= 0 && has_inv_space > 0) || (has_gun < 0 && has_inv_space > 1)) {
						string name_str(en->name);
						map<string, int>::iterator bit_it = bits_spent.find(name_str);
						if (bit_it != bits_spent.end()) {
							int spent = bit_it->second;
							if (bits_spent[name_str] < max_bits_per_game) {
								map<string, int>::iterator bit_it_s = bits_shield.find(name_str);
								if (bit_it_s != bits_shield.end()) {
									if (bit_it_s->second >= bits_per_shield) {
										for (int inv = 0; inv < 6; inv++) {
											if (pl->inventory[inv].item_id == UINT_MAX) {
												pl->inventory[inv].item_id = 51;
												pl->inventory[inv].item_param = 2;
												has_inv_space--;
												bits_shield[name_str] -= bits_per_shield;
												bits_spent[name_str] += bits_per_shield;
												buyfeed_add(&bf_rw, en->name, 51);
												//printf("bought shield, spent %i\n", bits_spent[name_str]);
												break;
											}
										}
									}
								}
							}
							if (bits_spent[name_str] < max_bits_per_game) {
								if ((has_gun >= 0 && has_inv_space > 0) || (has_gun < 0 && has_inv_space > 1)) {
									map<string, int>::iterator bit_it_b = bits_bandage.find(name_str);
									if (bit_it_b != bits_bandage.end()) {
										if (bit_it_b->second >= bits_per_bandage) {
											for (int inv = 0; inv < 6; inv++) {
												if (pl->inventory[inv].item_id == UINT_MAX) {
													pl->inventory[inv].item_id = 52;
													pl->inventory[inv].item_param = 5;
													has_inv_space--;
													bits_bandage[name_str] -= bits_per_bandage;
													bits_spent[name_str] += bits_per_bandage;
													buyfeed_add(&bf_rw, en->name, 52);
													//printf("bought bandage, spent %i\n", bits_spent[name_str]);
													break;
												}
											}
										}
									}
								}
							}
						}
					}

					//current position
					int gi = grid_get_index(bf_rw.data, gd.position_in_bf, { en->position[0], en->position[1], 0.0f });
					if (gi > -1) {
						int g_data_pos = bf_rw.data[gd.data_position_in_bf + 1 + gi];
						if (g_data_pos > 0) {
							int element_count = bf_rw.data[g_data_pos];
							for (int e = 0; e < element_count; e++) {
								unsigned int entity_id = bf_rw.data[g_data_pos + 1 + e];
								if (entity_id != pl->entity_id && entity_id < UINT_MAX) {
									struct entity* etc = &es[entity_id];
									if (etc->et == ET_ITEM) {
										if (etc->model_id == 50) { //colt
											if (has_gun < 0) {
												for (int inv = 0; inv < 6; inv++) {
													if (pl->inventory[inv].item_id == UINT_MAX) {
														//printf("picked up a gun\n");
														pl->inventory[inv].item_id = 50;
														pl->inventory[inv].item_param = 5;
														grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, etc->position, { item_models[0].model_scale, item_models[0].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&item_models[0]), entity_id);
														break;
													}
												}
											}
										}
										else if (etc->model_id == 51) { //shield
											if ((has_gun >= 0 && has_inv_space > 0) || (has_gun < 0 && has_inv_space > 1)) {
												for (int inv = 0; inv < 6; inv++) {
													if (pl->inventory[inv].item_id == UINT_MAX) {
														//printf("picked up shield\n");
														pl->inventory[inv].item_id = 51;
														pl->inventory[inv].item_param = 2;
														grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, etc->position, { item_models[1].model_scale, item_models[1].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&item_models[1]), entity_id);
														break;
													}
												}
											}
										}
										else if (etc->model_id == 52) { //bandage
											if ((has_gun >= 0 && has_inv_space > 0) || (has_gun < 0 && has_inv_space > 1)) {
												for (int inv = 0; inv < 6; inv++) {
													if (pl->inventory[inv].item_id == UINT_MAX) {
														//printf("picked up bandage\n");
														pl->inventory[inv].item_id = 52;
														pl->inventory[inv].item_param = 5;
														grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, etc->position, { item_models[2].model_scale, item_models[2].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&item_models[2]), entity_id);
														break;
													}
												}
											}
										}
									}
								}
							}
						}
					}
					float delta_x = 0.0f;
					float delta_y = 0.0f;

					if (storm_next_move_time(en->position, player_dist_per_tick) == 1.0f) {
						float dist = sqrtf((storm_to.x - en->position[0]) * (storm_to.x - en->position[0]) + (storm_to.y - en->position[1]) * (storm_to.y - en->position[1])) + 1e-5;

						delta_x = player_dist_per_tick * ((storm_to.x - en->position[0]) / dist);
						delta_y = player_dist_per_tick * ((storm_to.y - en->position[1]) / dist);
					}

					struct vector2<int> spiral_pos = { (int)en->position[0], (int)en->position[1] + 32 };
					struct vector2<int> spiral_dir[4] = { {0, 32}, { 32, 0 }, { 0, -32 }, { -32, 0 } };
					int					spiral_dir_idx = 1;
					struct vector2<int> spiral_dir_current = spiral_dir[spiral_dir_idx];
					int					spiral_steps_last = 1;
					int					spiral_steps = 1;
					int					spiral_steps_counter = 0;

					while (spiral_steps < 10) {
						//process grid
						int gi = grid_get_index(bf_rw.data, gd.position_in_bf, { (float)spiral_pos[0], (float)spiral_pos[1], 0.0f });
						if (gi > -1) {
							int g_data_pos = bf_rw.data[gd.data_position_in_bf + 1 + gi];
							if (g_data_pos > 0) {
								float dist = sqrtf((spiral_pos[0] - (int)en->position[0]) * (spiral_pos[0] - (int)en->position[0]) + (spiral_pos[1] - (int)en->position[1]) * (spiral_pos[1] - (int)en->position[1])) + 1e-5;
								int element_count = bf_rw.data[g_data_pos];
								for (int e = 0; e < element_count; e++) {
									unsigned int entity_id = bf_rw.data[g_data_pos + 1 + e];
									if (entity_id != pl->entity_id && entity_id < UINT_MAX) {
										struct entity* etc = &es[entity_id];
										if (etc->et == ET_ITEM && delta_x == 0 && delta_y == 0) {
											if (!storm_is_in({ (float)spiral_pos[0], (float)spiral_pos[1], 0.0f })) {
												if (etc->model_id == 50) { //colt
													if (has_gun < 0) {
														delta_x = player_dist_per_tick * ((spiral_pos[0] - (int)en->position[0]) / dist);
														delta_y = player_dist_per_tick * ((spiral_pos[1] - (int)en->position[1]) / dist);
													}
												}
												else if (etc->model_id == 51) { // shield
													if ((has_gun >= 0 && has_inv_space > 0) || (has_gun < 0 && has_inv_space > 1)) {
														delta_x = player_dist_per_tick * ((spiral_pos[0] - (int)en->position[0]) / dist);
														delta_y = player_dist_per_tick * ((spiral_pos[1] - (int)en->position[1]) / dist);
													}
												}
												else if (etc->model_id == 52) { // bandage
													if ((has_gun >= 0 && has_inv_space > 0) || (has_gun < 0 && has_inv_space > 1)) {
														delta_x = player_dist_per_tick * ((spiral_pos[0] - (int)en->position[0]) / dist);
														delta_y = player_dist_per_tick * ((spiral_pos[1] - (int)en->position[1]) / dist);
													}
												}
											}
										}
										else if (etc->et == ET_PLAYER && has_gun >= 0) {
											if (dist / 32 < 5 && pl->inventory[has_gun].item_param % 15 == 0) {
												if (players[etc->name].health > 0) {
													pl->inventory[has_gun].item_param++;
													float hit = (rand() / (float)RAND_MAX);
													//printf("player: %s shoots at %s", pl->name, etc->name);
													if (hit < 0.8) {
														if (players[etc->name].shield > 0) {
															players[etc->name].shield -= 10;
														}
														else {
															players[etc->name].health -= 10;
														}
														pl->damage_dealt += 10;
														//printf(" hit");
														if (players[etc->name].health <= 0) {
															pl->kills++;
															//printf(" & kill");
															killfeed_add(&bf_rw, pl->name, players[etc->name].name);
															leaderboard_add(&bf_rw, players[etc->name].name, players[etc->name].damage_dealt, players[etc->name].kills, pl->name);
														}
													}
													//printf("\n");
												}
											}
										}
									}
								}
							}
						}
						//spiral_pos_next
						spiral_pos = { spiral_pos[0] + spiral_dir_current[0], spiral_pos[1] + spiral_dir_current[1] };
						spiral_steps--;
						if (spiral_steps == 0) {
							spiral_steps_counter++;
							spiral_dir_current = spiral_dir[(spiral_dir_idx + 1) % 4];
							spiral_dir_idx++;
							spiral_steps = spiral_steps_last;
							if (spiral_steps_counter == 2) {
								spiral_steps++;
								spiral_steps_counter = 0;
							}
							spiral_steps_last = spiral_steps;
						}
					}
					int target_orientation = en->orientation;
					if (delta_x == 0 && delta_y == 0) {
						if (rand() / (float)RAND_MAX < 0.1) {
							int fac = 1;
							if (rand() / (float)RAND_MAX < 0.20) fac = -1;
							en->orientation += 3 * fac;
						}
						delta_x = -1 * player_dist_per_tick * cos(3.1415 / 180.0f * (en->orientation + 90));
						delta_y = -1 * player_dist_per_tick * sin(3.1415 / 180.0f * (en->orientation - 90));

						if (en->position[0] + delta_x < 32) {
							delta_x = 0.0f;
							delta_y = 0.0f;
							target_orientation = 45 + (int)(rand() / (float)RAND_MAX * 90);
							en->orientation = target_orientation;
						}
						if (en->position[0] + delta_x >= gm.map_dimensions[0] - 32) {
							delta_x = 0.0f;
							delta_y = 0.0f;
							target_orientation = 315 - (int)(rand() / (float)RAND_MAX * 90);
							en->orientation = target_orientation;
						}
						if (en->position[1] + delta_y < 32) {
							delta_x = 0.0f;
							delta_y = 0.0f;
							target_orientation = 315 + (int)(rand() / (float)RAND_MAX * 90);
							en->orientation = target_orientation;
						}
						if (en->position[1] + delta_y >= gm.map_dimensions[1] - 32) {
							delta_x = 0.0f;
							delta_y = 0.0f;
							target_orientation = 135 + (int)(rand() / (float)RAND_MAX * 90);
							en->orientation = target_orientation;
						}

						if (storm_is_in({ en->position[0] + delta_x, en->position[1] + delta_y, 0.0f })) {
							delta_x = 0.0f;
							delta_y = 0.0f;
							target_orientation += 3;
						}
					}
					else {
						target_orientation = (int)roundf(atan2(-delta_y, delta_x) * (180 / 3.1415f)) + 90;
					}
					if (target_orientation < 0) {
						target_orientation += 360;
					}
					if (abs(((int)en->orientation - target_orientation) % 360) > orientation_change_per_tick) {
						if (target_orientation > en->orientation) {
							if (abs(en->orientation + 360 - target_orientation) < abs(target_orientation - en->orientation)) {
								en->orientation -= orientation_change_per_tick;
							}
							else {
								en->orientation += orientation_change_per_tick;
							}
						}
						else {
							if (abs(en->orientation - (target_orientation + 360)) < abs(en->orientation - target_orientation)) {
								en->orientation += orientation_change_per_tick;
							}
							else {
								en->orientation -= orientation_change_per_tick;
							}
						}
						if (en->orientation < 0) en->orientation += 360;
					}
					else {
						en->orientation = target_orientation;
						//object itself
						grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, en->position, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&player_models[PT_HOPE]), pl->entity_id);
						//object text
						grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1] - 32.0f - 3, en->position[2] - 0.0f }, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { en->name_len * 32.0f + 32.0f + 3, 96.0f + 3, 0 }, pl->entity_id);
						//object inventory
						grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1], en->position[2] - 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 32.0f, 32.0f * 6, 0 }, pl->entity_id);

						en->force[0] = delta_x;
						en->force[1] = delta_y;

						float val_1 = exp(1 / (pow(en->force[0] - en->velocity[0], 2.0) + 1)) / exp(1);
						float val_1b = 1 - val_1;

						float val_2 = exp(1 / (pow(en->force[1] - en->velocity[1], 2.0) + 1)) / exp(1);
						float val_2b = 1 - val_2;

						float a = en->velocity[0] * val_1b + en->force[0] * val_1;
						float b = en->velocity[1] * val_2b + en->force[1] * val_2;

						en->velocity[0] = (a);
						en->velocity[1] = (b);

						en->position[0] += en->velocity[0];
						en->position[1] += en->velocity[1];

						//object itself
						grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, en->position, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&player_models[PT_HOPE]), pl->entity_id);
						es = (struct entity*) & bf_rw.data[entities_position];
						en = (struct entity*) & es[pl->entity_id];
						//object text
						grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1] - 32.0f - 3, en->position[2] - 0.0f }, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { en->name_len * 32.0f + 32.0f + 3, 96.0f + 3, 0 }, pl->entity_id);
						es = (struct entity*) & bf_rw.data[entities_position];
						en = (struct entity*) & es[pl->entity_id];
						//object inventory
						grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1], en->position[2] - 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 32.0f, 32.0f * 6, 0 }, pl->entity_id);
						es = (struct entity*) & bf_rw.data[entities_position];
						en = (struct entity*) & es[pl->entity_id];
					}
					en->params[0] = (char)players_it->second.health;
					en->params[1] = (char)players_it->second.shield;
					int* params = (int*)&en->params;
					int params_pos = 1;
					for (int ip = 0; ip < 6; ip++) {
						params[params_pos++] = players_it->second.inventory[ip].item_id;
						params[params_pos++] = players_it->second.inventory[ip].item_param;
					}
				}
			}
			players_it++;
		}
		players_it = players.begin();
		struct entity* es = (struct entity*) & bf_rw.data[entities_position];
		while (players_it != players.end()) {
			struct player* pl = &players_it->second;
			if (pl->kills > top_kills) {
				ui_textfield_set_value(&bf_rw, "ingame_menu", "top_kills", pl->name);
				ui_textfield_set_int(&bf_rw, "ingame_menu", "top_kills_nr", pl->kills);
				top_kills = pl->kills;
			}
			if (pl->damage_dealt > top_damage) {
				ui_textfield_set_value(&bf_rw, "ingame_menu", "top_damage", pl->name);
				ui_textfield_set_int(&bf_rw, "ingame_menu", "top_damage_nr", pl->damage_dealt);
				top_damage = pl->damage_dealt;
			}
			if (pl->alive) {
				if (pl->health <= 0) {
					struct entity* en = &es[pl->entity_id];
					pl->alive = false;
					//object itself
					grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, en->position, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&player_models[PT_HOPE]), pl->entity_id);
					//object text
					grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1] - 32.0f - 3, en->position[2] - 0.0f }, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { en->name_len * 32.0f + 32.0f + 3, 96.0f + 3, 0 }, pl->entity_id);
					//object inventory
					grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1], en->position[2] - 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 32.0f, 32.0f * 6, 0 }, pl->entity_id);
					pl->entity_id = UINT_MAX;
				}
			}
			players_it++;
		}

		es = (struct entity*) & bf_rw.data[entities_position];
		for (int e = 0; e < entities.size(); e++) {
			struct entity* en = &es[e];
			if (en->et == ET_ITEM) {
				en->orientation += 3;
			}
		}
		bit_field_invalidate_bulk(&bf_rw, entities_position, entities_size_in_bf);

		storm_next(&bf_assets, &bf_rw);

		if (game_ticks % 30 == 0) {
				twitch_update_bits();
				struct entity* es = (struct entity*) & bf_rw.data[entities_position];
				map<string, struct player>::iterator players_it = players.begin();
				while (players_it != players.end()) {
					//for (int i = 0; i < players.size(); i++) {
					struct player* pl = &players_it->second;
					if (pl->alive) {
						if (kill_count == players.size() - 1) {
							ui_textfield_set_value(&bf_rw, "ingame_menu", "top_placement", pl->name);
							char tmp[2];
							tmp[0] = '^';
							tmp[1] = '\0';
							leaderboard_add(&bf_rw, pl->name, pl->damage_dealt, pl->kills, tmp);
							pl->alive = false;
							game_ticks_target = 35;
							ui_set_active("ingame_menu");
						}
						if (pl->entity_id < UINT_MAX) {
							//printf("%i ", pl->entity_id);
							struct entity* en = &es[pl->entity_id];
							if (storm_is_in(en->position)) {
								pl->health -= storm_phase_dps[storm_phase_current];
							}
							if (pl->health <= 0) {
								pl->alive = false;
								killfeed_add(&bf_rw, NULL, pl->name, true);
								char tmp[2];
								tmp[0] = ';';
								tmp[1] = '\0';
								leaderboard_add(&bf_rw, pl->name, pl->damage_dealt, pl->kills, tmp);
								//object itself
								grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, en->position, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&player_models[PT_HOPE]), pl->entity_id);
								//object text
								grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1] - 32.0f - 3, en->position[2] - 0.0f }, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { en->name_len * 32.0f + 32.0f + 3, 96.0f + 3, 0 }, pl->entity_id);
								//object inventory
								grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1], en->position[2] - 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 32.0f, 32.0f * 6, 0 }, pl->entity_id);
								pl->entity_id = UINT_MAX;
							}
						}
					}
					players_it++;
				}
			}
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

	//reset bits
	bits_spent.clear();
	bits_shield.clear();
	bits_bandage.clear();

	//reset playerlist
	playerlist_reset(&bf_rw);

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

	game_ticks_target = 30;
}