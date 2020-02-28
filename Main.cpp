#include "Main.hpp"

#include <sstream>

#include "BitField.hpp"
#include "AssetLoader.hpp"
#include "UI.hpp"
#include "Map.hpp"
#include "Grid.hpp"
#include "Camera.hpp"
#include "SDLShow.hpp"
#include "Storm.hpp"
#include "Entity.hpp"
#include "TwitchIntegration.hpp"
#include "KillFeed.hpp"
#include "Playerlist.hpp"
#include "Leaderboard.hpp"
#include "Util.hpp"


#include "time.h"
#include "math.h"

#include <ctime>
#include <chrono>
#include <thread>
#include <random>
#include <map>
#include <utility>



using namespace std;

struct grid gd;

struct bit_field bf_assets;
struct bit_field bf_rw;
struct bit_field bf_output;

struct vector3<float> camera;
struct vector2<unsigned int> resolution;

struct vector2<unsigned int> map_dimensions;

int	max_bits_per_game = 0;
int bits_per_shield = 0;
int bits_per_bandage = 0;
map<string, int>				bits_spent;
map<string, int>				bits_shield;
map<string, int>				bits_bandage;

int target_ticks_per_second = 30;
unsigned int tick_counter = 0;

bool game_started = false;
bool irc_started = false;
bool running = true;

int top_kills = 0;
int top_damage = 0;

void start_game() {
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
	players_upload(&bf_rw);

	killfeed_init(&bf_rw);

	ui_value_as_config(&bf_rw, "ingame_overlay", "killfeed", 0, kill_feed_pos);

	leaderboard_init(&bf_rw);

	//spawn weapons
	for (int i = 0; i < players.size(); i++) {
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
	for (int i = 0; i < players.size()/2; i++) {
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
	for (int i = 0; i < players.size() / 2; i++) {
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

	ui_set_active("ingame_overlay");
	game_started = true;
}

int main(int argc, char** argv) {
	resolution = {1920, 1080};
	unsigned int output_size = resolution[0] * resolution[1] * 4;
	unsigned int output_size_in_bf = (int)ceilf(output_size / (float) sizeof(unsigned int));
	unsigned int output_position;

	srand(time(nullptr));

	//bit field uploaded once
	bit_field_init(&bf_assets, 16, 1024);
	bit_field_register_device(&bf_assets, 0);

	//uploaded & downloaded per frame
	bit_field_init(&bf_rw, 16, 1024);
	bit_field_register_device(&bf_rw, 0);

	/////////////////
	// -- ASSETS --//
	/////////////////
	
	// -- UI -- //

	ui_init(&bf_assets, &bf_rw);
	ui_set_active("main_menu");

	// -- LOAD SETTINGS -- //
	vector<pair<string, string>> kv_pairs = get_cfg_key_value_pairs("./", "settings.cfg");
	for (int i = 0; i < kv_pairs.size(); i++) {
		if (kv_pairs[i].first == "ui") {
			string ui_name = kv_pairs[i].second;
			ui_name = trim(ui_name);
			string ui_field = "";
			string ui_value = "";
			if (kv_pairs[i + 1].first == kv_pairs[i].second + "_field") {
				ui_field = kv_pairs[i + 1].second;
				ui_field = trim(ui_field);
			}
			if (kv_pairs[i + 2].first == kv_pairs[i].second + "_value") {
				ui_value = kv_pairs[i + 2].second;
				ui_value = trim(ui_value);
			}
			if (ui_field != "" && ui_value != "") {
				printf("loading from cfg for %s field %s value %s\n", ui_name.c_str(), ui_field.c_str(), ui_value.c_str());
				ui_textfield_set_value(&bf_rw, ui_name, ui_field, ui_value.c_str());
			}
		}
	}

	// -- MAPS -- //

	//load list of available maps
	map_list_init();
	
	//load first map
	map_load(&bf_assets, game_maps[0]);
	

	// -- MODELS -- //

	//load player models
	player_models_init(&bf_assets);

	//load item models
	item_models_init(&bf_assets);
	
	//uploading assets to gpu
	bit_field_update_device(&bf_assets, 0);


	///////////////////////
	// -- RW BITFIELD -- //
	///////////////////////

	grid_init(&bf_rw, &gd, struct vector3<float>((float)gm.map_dimensions[0], (float)gm.map_dimensions[1], 1.0f), struct vector3<float>(32.0f, 32.0f, 1.0f), struct vector3<float>(0, 0, 0));

	map_add_static_assets(&bf_assets, &bf_rw, &gd);
	

	///////////////////////////
	// -- OUTPUT BITFIELD -- //
	///////////////////////////

	//downloaded per frame
	bit_field_init(&bf_output, 16, 1024);
	output_position = bit_field_add_bulk_zero(&bf_output, output_size_in_bf)+1;
	bit_field_register_device(&bf_output, 0);

	camera = { 0.0f, 0.0f, 1.0f };

	storm_init(&bf_assets, &bf_rw);
	//players_ptr = (struct player*) & bf_rw.data[players_position];
	
	vector<unsigned int> camera_crop;
	camera_crop.push_back(0);
	camera_crop.push_back(0);
	camera_crop.push_back(0);
	camera_crop.push_back(0);
	camera_get_crop(camera_crop);

	vector<unsigned int> camera_crop_tmp;
	camera_crop_tmp.push_back(0);
	camera_crop_tmp.push_back(0);
	camera_crop_tmp.push_back(0);
	camera_crop_tmp.push_back(0);
	struct vector3<float> camera_tmp = camera;
	struct vector2<unsigned int> mouse_position;

	sdl_show_window();
	SDL_Event sdl_event;

	float sensitivity_z = 0.1f;
	float sensitivity_xy = 0.5f;
	float sensitivity_zoom_ratio = 1.0f;

	int fps = 0;
	double sec = 0;

	int frame_balancing = (int)floorf(1000.0f/(float)target_ticks_per_second);
	long tf = clock();

	while (running) {
		long tf_l = clock();

		while (SDL_PollEvent(&sdl_event) != 0) {
			if (game_started) {
				if (sdl_event.type == SDL_KEYDOWN && sdl_event.key.keysym.sym == SDLK_ESCAPE) {
					ui_set_active("ingame_menu");
				}

				float camera_delta_z = 0.0f;
				if (sdl_event.type == SDL_MOUSEWHEEL) {
					if (!ui_process_scroll(&bf_rw, mouse_position[0], mouse_position[1], sdl_event.wheel.y)) {
						camera_delta_z -= sdl_event.wheel.y * sensitivity_z;
						bool camera_z_has_moved = false;
						float camera_z = camera[2];
						camera_crop_tmp = camera_crop;
						camera_move(struct vector3<float>(0.0f, 0.0f, camera_delta_z));
						camera_get_crop(camera_crop);
						if (camera_z != camera[2]) {
							camera_move(struct vector3<float>(camera_crop_tmp[0] - camera_crop[0] + mouse_position[0] * (camera_z - camera[2]), 0.0f, 0.0f));
							camera_get_crop(camera_crop);
							camera_move(struct vector3<float>(0.0f, camera_crop_tmp[2] - camera_crop[2] + mouse_position[1] * (camera_z - camera[2]), 0.0f));
							camera_get_crop(camera_crop);
						}
					}
				}

				if (sdl_event.type == SDL_MOUSEMOTION && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_RIGHT)) {
					float zoom_sensitivity = sensitivity_xy * camera[2] * sensitivity_zoom_ratio;
					if (zoom_sensitivity < 0.2f) zoom_sensitivity = 0.2f;
					camera_move(struct vector3<float>(-sdl_event.motion.xrel * zoom_sensitivity, -sdl_event.motion.yrel * zoom_sensitivity, 0.0f));
					camera_get_crop(camera_crop);
				}
			} else {
				if (sdl_event.type == SDL_MOUSEWHEEL) {
					ui_process_scroll(&bf_rw, mouse_position[0], mouse_position[1], sdl_event.wheel.y);
				}
			}

			if (sdl_event.type == SDL_KEYDOWN) {
				ui_process_keys(&bf_rw, mouse_position[0], mouse_position[1], sdl_event.key.keysym.sym);
			}

			if (sdl_event.type == SDL_MOUSEMOTION) {
				mouse_position[0] = sdl_event.motion.x;
				mouse_position[1] = sdl_event.motion.y;
			}

			if (ui_active != "") {
				if (sdl_event.type == SDL_MOUSEBUTTONUP && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_LEFT)) {
					//printf("clicking %i %i\n", mouse_position[0], mouse_position[1]);
					ui_process_click(mouse_position[0], mouse_position[1]);
				}
			}
		}

		if (game_started) {
			struct entity* es = (struct entity*) & bf_rw.data[entities_position];
			for (int e = 0; e < entities.size(); e++) {
				struct entity* en = &es[e];
				if (en->et == ET_ITEM) {
					en->orientation += 3;
				}
			}
			bit_field_invalidate_bulk(&bf_rw, entities_position, entities_size_in_bf);
			bit_field_update_device(&bf_rw, 0);

			launch_draw_map(bf_assets.device_data[0], gm.map_zoom_level_count, gm.map_zoom_center_z, gm.map_zoom_level_offsets_position, gm.map_positions, resolution[0], resolution[1], 4, camera_crop[0], camera_crop[1], camera_crop[2], camera_crop[3], bf_output.device_data[0], output_position, 1920, 1080);

			launch_draw_entities_kernel(bf_assets.device_data[0], player_models_position, item_models_position, map_models_position, ui_fonts_position, bf_rw.device_data[0], entities_position, gd.position_in_bf, gd.data_position_in_bf,
				bf_output.device_data[0], output_position, 1920, 1080, 4, camera_crop[0], camera_crop[2], camera[2], mouse_position, tick_counter);
		
			launch_draw_storm_kernel(bf_output.device_data[0], output_position, resolution[0], resolution[1], 4, camera_crop[0], camera_crop[2], camera[2], storm_current, storm_to, 50, { 45, 0, 100 });

			storm_next(&bf_assets, &bf_rw);
		} else {
			bit_field_update_device(&bf_rw, 0);
		}

		if (ui_active != "") {
			struct ui u = uis[ui_active];
			if (!game_started) {
				unsigned char* output_frame = (unsigned char *)&bf_output.data[output_position];
				memset(output_frame, 0, output_size);
				bit_field_invalidate_bulk(&bf_output, output_position, output_size_in_bf);
				bit_field_update_device(&bf_output, 0);
			}
			launch_draw_ui_kernel(bf_assets.device_data[0], u.background_position, ui_fonts_position, bf_output.device_data[0], output_position, resolution[0], resolution[1], 4, bf_rw.device_data[0], tick_counter);
		}

		bit_field_update_host(&bf_output, 0);

		sdl_update_frame((Uint32*)&bf_output.data[output_position]);

		if (game_started) {
			struct entity* es = (struct entity*) & bf_rw.data[entities_position];
			map<string, struct player>::iterator players_it = players.begin();
			float player_dist_per_tick = 1 / 5.0f;
			int orientation_change_per_tick = 3;
			while (players_it != players.end()) {
				struct player* pl = &players_it->second;
				if (pl->alive) {
					if (pl->entity_id < UINT_MAX) {
						struct entity* en = &es[pl->entity_id];
						int has_inv_space = 0;
						int has_gun = -1;
						for (int inv = 0; inv < 6; inv++) {
							if (pl->inventory[inv].item_id == UINT_MAX) {
								has_inv_space++;
							} else if (pl->inventory[inv].item_id == 50) {
								has_gun = inv;
								if (pl->inventory[inv].item_param % 15 != 0) {
									pl->inventory[inv].item_param++;
								}
							} else if (pl->inventory[inv].item_id == 51) {
								if (pl->shield <= 75 && pl->inventory[inv].item_param != 0) {
									//printf("shiedling\n");
									pl->shield += 25;
									pl->inventory[inv].item_param--;
								}
								if (pl->inventory[inv].item_param == 0) {
									pl->inventory[inv].item_id = UINT_MAX;
									has_inv_space++;
								}
							} else if (pl->inventory[inv].item_id == 52) {
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
											} else if (etc->model_id == 51) { //shield
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
											} else if (etc->model_id == 52) { //bandage
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
													} else if (etc->model_id == 51) { // shield
														if ((has_gun >= 0 && has_inv_space > 0) || (has_gun < 0 && has_inv_space > 1)) {
															delta_x = player_dist_per_tick * ((spiral_pos[0] - (int)en->position[0]) / dist);
															delta_y = player_dist_per_tick * ((spiral_pos[1] - (int)en->position[1]) / dist);
														}
													} else if (etc->model_id == 52) { // bandage
														if ((has_gun >= 0 && has_inv_space > 0) || (has_gun < 0 && has_inv_space > 1)) {
															delta_x = player_dist_per_tick * ((spiral_pos[0] - (int)en->position[0]) / dist);
															delta_y = player_dist_per_tick * ((spiral_pos[1] - (int)en->position[1]) / dist);
														}
													}
												}
											} else if (etc->et == ET_PLAYER && has_gun >= 0) {
												if (dist / 32 < 5 && pl->inventory[has_gun].item_param % 15 == 0) {
													if (players[etc->name].health > 0) {
														pl->inventory[has_gun].item_param++;
														float hit = (rand() / (float)RAND_MAX);
														//printf("player: %s shoots at %s", pl->name, etc->name);
														if (hit < 0.8) {
															if (players[etc->name].shield > 0) {
																players[etc->name].shield -= 10;
															} else {
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
							delta_x = -1*player_dist_per_tick*cos(3.1415/180.0f * (en->orientation+90));
							delta_y = -1*player_dist_per_tick*sin(3.1415/180.0f * (en->orientation-90));

							if (en->position[0] + delta_x < 32) {
								delta_x = 0.0f;
								delta_y = 0.0f;
								target_orientation = 45 + (int)(rand()/(float) RAND_MAX * 90);
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
								target_orientation+=3;
							}
						} else {
							target_orientation = (int)roundf(atan2(-delta_y, delta_x) * (180 / 3.1415f)) + 90;
						}
						if (target_orientation < 0) {
							target_orientation += 360;
						}
						if (abs(((int)en->orientation - target_orientation) % 360) > orientation_change_per_tick) {
							if (target_orientation > en->orientation) {
								if (abs(en->orientation + 360 - target_orientation) < abs(target_orientation - en->orientation)) {
									en->orientation -= orientation_change_per_tick;
								} else {
									en->orientation += orientation_change_per_tick;
								}
							} else {
								if (abs(en->orientation - (target_orientation + 360)) < abs(en->orientation - target_orientation)) {
									en->orientation += orientation_change_per_tick;
								} else {
									en->orientation -= orientation_change_per_tick;
								}
							}
							if (en->orientation < 0) en->orientation += 360;
						} else {
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
							//object text
							grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1] - 32.0f - 3, en->position[2] - 0.0f }, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { en->name_len * 32.0f + 32.0f + 3, 96.0f + 3, 0 }, pl->entity_id);
							//object inventory
							grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1], en->position[2] - 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f }, { 32.0f, 32.0f*6, 0 }, pl->entity_id);
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
		}

		if (tick_counter % 30 == 0) {
			if (game_started) {
				twitch_update_bits();
				//struct player* players_ptr = (struct player*) &bf_rw.data[players_position];
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
			} else {
				if (ui_active == "lobby" && irc_started) {
					twitch_update_players(&bf_rw);
					ui_textfield_set_int(&bf_rw, "lobby", "playercount", players.size());
				}
				if (ui_active == "lobby" && !irc_started) {
					string twitch_name = "";
					for (int uc = 0; uc < uis["settings"].ui_elements.size(); uc++) {
						if (uis["settings"].ui_elements[uc].name == "twitch_name") {
							for (int ca = 0; ca < 50; ca++) {
								if (uis["settings"].ui_elements[uc].value[ca] != '\0') {
									twitch_name += uis["settings"].ui_elements[uc].value[ca];
								}
							}
						}
						if (uis["settings"].ui_elements[uc].name == "bits_bandage") {
							string bits_ = "";
							for (int ca = 0; ca < 50; ca++) {
								if (uis["settings"].ui_elements[uc].value[ca] != '\0') {
									bits_ += uis["settings"].ui_elements[uc].value[ca];
								}
							}
							bits_per_bandage = stoi(bits_);
						}
						if (uis["settings"].ui_elements[uc].name == "bits_shield") {
							string bits_ = "";
							for (int ca = 0; ca < 50; ca++) {
								if (uis["settings"].ui_elements[uc].value[ca] != '\0') {
									bits_ += uis["settings"].ui_elements[uc].value[ca];
								}
							}
							bits_per_shield = stoi(bits_);
						}
						if (uis["settings"].ui_elements[uc].name == "bits_game") {
							string bits_ = "";
							for (int ca = 0; ca < 50; ca++) {
								if (uis["settings"].ui_elements[uc].value[ca] != '\0') {
									bits_ += uis["settings"].ui_elements[uc].value[ca];
								}
							}
							max_bits_per_game = stoi(bits_);
						}
						if (uis["settings"].ui_elements[uc].name == "max_players") {
							string max_players_ = "";
							for (int ca = 0; ca < 50; ca++) {
								if (uis["settings"].ui_elements[uc].value[ca] != '\0') {
									max_players_ += uis["settings"].ui_elements[uc].value[ca];
								}
							}
							players_max = stoi(max_players_);
							printf("setting max_players to %i\n", players_max);
						}
					}
					if (twitch_name != "") {
						playerlist_init(&bf_rw);

						ui_value_as_config(&bf_rw, "lobby", "players", 0, playerlist_pos);

						stringstream ss;
						ss << rand();
						string cache_dir = "cache\\" + ss.str();
						twitch_launch_irc(cache_dir, twitch_name);
						irc_started = true;
					}
				}
			}
		}

		long tf_3 = clock();
		long tf_ = tf_3 - tf;
		sec += ((double)tf_ / CLOCKS_PER_SEC);
		tf = clock();
		fps++;

		if (sec >= 1) {
			printf("main fps: %d\r\n", fps);
			//printf("main ft: %f, main ft_l: %f\r\n", tf_ / (double)CLOCKS_PER_SEC, (tf_3 - tf_l) / (double)CLOCKS_PER_SEC);
			double frame_time = (tf_3 - tf_l) / (double)CLOCKS_PER_SEC;
			//printf("frame_time %f, target_frame_time %f\n", frame_time, 1000.0f / (float)target_ticks_per_second / 1000.0f);
			//printf("going for: %f\n", floorf((1000.0f / (float)target_ticks_per_second / 1000.0f - frame_time) * 1000.0f));
			if (frame_time < 1000.0f/(float)target_ticks_per_second / 1000.0f) {
				frame_balancing = (int)floorf((1000.0f / (float)target_ticks_per_second / 1000.0f - frame_time)*1000.0f);
			} else {
				frame_balancing = 0;
			}
			//printf("frame_balancing %i\n", frame_balancing);
			/*if (fps > target_ticks_per_second+2) {
				frame_balancing++;
			} else if (fps < target_ticks_per_second && frame_balancing > 0) {
				frame_balancing--;
			}*/
			sec = 0;
			fps = 0;
		}
		if (frame_balancing > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(frame_balancing));
		}

		tick_counter++;
	}

	string s_fields[5] = { "twitch_name", "bits_bandage", "bits_shield", "bits_game", "max_players" };
	ui_save_fields_to_file(&bf_rw, "settings", s_fields, 5, "./", "settings.cfg");

	twitch_terminate_irc();

	return 0;
}