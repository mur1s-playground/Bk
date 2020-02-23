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


#include "time.h"
#include "math.h"

#include <ctime>
#include <chrono>
#include <thread>
#include <random>



using namespace std;

struct grid gd;

struct bit_field bf_assets;
struct bit_field bf_rw;
struct bit_field bf_output;

struct vector3<float> camera;
struct vector2<unsigned int> resolution;

struct vector2<unsigned int> map_dimensions;

int target_ticks_per_second = 30;
unsigned int tick_counter = 0;

bool game_started = false;
bool irc_started = false;
bool running = true;

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
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { cur_e->position[0] - 32.0f - 3, cur_e->position[1] - 32.0f - 3, cur_e->position[2] - 0.0f }, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { cur_e->name_len*32.0f+32.0f+3, 96.0f+3, 0 }, entities.size() - 1);
		pl_it->second.entity_id = entities.size() - 1;

		i++;
		pl_it++;
	}

	bit_field_update_device(&bf_assets, 0);
	players_upload(&bf_rw);

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

				if (sdl_event.type == SDL_MOUSEMOTION && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_RIGHT)) {
					float zoom_sensitivity = sensitivity_xy * camera[2] * sensitivity_zoom_ratio;
					if (zoom_sensitivity < 0.2f) zoom_sensitivity = 0.2f;
					camera_move(struct vector3<float>(-sdl_event.motion.xrel * zoom_sensitivity, -sdl_event.motion.yrel * zoom_sensitivity, 0.0f));
					camera_get_crop(camera_crop);
				}
			}

			if (sdl_event.type == SDL_KEYDOWN) {
				ui_process_keys(sdl_event.key.keysym.sym, &bf_rw);
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
				bf_output.device_data[0], output_position, 1920, 1080, 4, camera_crop[0], camera_crop[2], camera[2], tick_counter);
		
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
			launch_draw_ui_kernel(bf_assets.device_data[0], u.background_position, ui_fonts_position, bf_output.device_data[0], output_position, resolution[0], resolution[1], 4, bf_rw.device_data[0]);
		}

		bit_field_update_host(&bf_output, 0);

		sdl_update_frame((Uint32*)&bf_output.data[output_position]);

		long tf_3 = clock();
		long tf_ = tf_3 - tf;
		sec += ((double)tf_ / CLOCKS_PER_SEC);
		tf = clock();
		fps++;

		if (sec >= 1.0) {
			if (game_started) {
				//struct player* players_ptr = (struct player*) &bf_rw.data[players_position];
				struct entity* es = (struct entity*) & bf_rw.data[entities_position];
				map<string, struct player>::iterator players_it = players.begin();
				while (players_it != players.end()) {
					//for (int i = 0; i < players.size(); i++) {
					struct player* pl = &players_it->second;
					if (pl->entity_id < UINT_MAX) {
						//printf("%i ", pl->entity_id);
						struct entity* en = &es[pl->entity_id];
						if (storm_is_in(en->position)) {
							pl->health -= storm_phase_dps[storm_phase_current];
						}
						if (pl->health < 0) {
							//object itself
							grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, en->position, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&player_models[PT_HOPE]), pl->entity_id);
							//object text
							grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, { en->position[0] - 32.0f - 3, en->position[1] - 32.0f - 3, en->position[2] - 0.0f }, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { en->name_len * 32.0f + 32.0f + 3, 96.0f + 3, 0 }, pl->entity_id);
							pl->entity_id = UINT_MAX;
						}
					}
					players_it++;
				}
			} else {
				if (ui_active == "lobby" && irc_started) {
					twitch_update_players();
					printf("player_count %i\n", players.size());
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
							break;
						}
					}
					if (twitch_name != "") {
						stringstream ss;
						ss << rand();
						string cache_dir = "cache\\" + ss.str();
						twitch_launch_irc(cache_dir, twitch_name);
						irc_started = true;
					}
				}
			}
			
			printf("main fps: %d\r\n", fps);
			printf("main ft: %f, main ft_l: %f\r\n", tf_ / (double)CLOCKS_PER_SEC, (tf_3 - tf_l) / (double)CLOCKS_PER_SEC);
			if (fps > target_ticks_per_second+2) {
				frame_balancing++;
			} else if (fps < target_ticks_per_second && frame_balancing > 0) {
				frame_balancing--;
			}
			sec = 0;
			fps = 0;
		}
		if (frame_balancing > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(frame_balancing));
		}

		tick_counter++;
	}

	twitch_terminate_irc();

	return 0;
}