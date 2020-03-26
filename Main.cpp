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
#include "Buyfeed.hpp"
#include "Playerlist.hpp"
#include "Leaderboard.hpp"
#include "Util.hpp"
#include "Game.hpp"
#include "MapEditor.hpp"


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
struct bit_field bf_map;
struct bit_field bf_rw;
struct bit_field bf_output;

struct vector2<unsigned int> resolution;

struct vector2<unsigned int> map_dimensions;

int	max_bits_per_game = 0;
int bits_per_shield = 0;
int bits_per_bandage = 0;
map<string, int>				bits_spent;
map<string, int>				bits_shield;
map<string, int>				bits_bandage;

int ui_ticks_per_second = 60;
unsigned int ui_tick_counter = 0;

bool map_editor = false;
bool map_editor_update_assets = false;

bool irc_started = false;
bool running = true;

int top_kills = 0;
int top_damage = 0;

struct vector2<unsigned int> mouse_position;

int main(int argc, char** argv) {
	resolution = {1920, 1080};
	unsigned int output_size = resolution[0] * resolution[1] * 4;
	unsigned int output_size_in_bf = (int)ceilf(output_size / (float) sizeof(unsigned int));
	unsigned int output_position;

	srand(time(nullptr));

	//bit field uploaded "once"
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
	bit_field_update_device(&bf_assets, 0);
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
	map_list_init(&bf_rw);
	
	// -- MODELS -- //

	//load player models
	player_models_init(&bf_assets);

	//load item models
	item_models_init(&bf_assets);

	bit_field_update_device(&bf_assets, 0);
	
	///////////////////////////
	// -- OUTPUT BITFIELD -- //
	///////////////////////////

	//downloaded per frame
	bit_field_init(&bf_output, 16, 1024);
	output_position = bit_field_add_bulk_zero(&bf_output, output_size_in_bf)+1;
	bit_field_register_device(&bf_output, 0);

	camera_init();

	sdl_show_window();
	SDL_Event sdl_event;

	float sensitivity_z = 0.1f;
	float sensitivity_xy = 0.5f;
	float sensitivity_zoom_ratio = 1.0f;

	int frame_balancing = (int)floorf(1000.0f/(float)ui_ticks_per_second);
	long tf = clock();

	if (argc >= 3) {
		if (string(argv[1]) == "--editor") {
			map_editor = true;
			ui_textfield_set_value(&bf_rw, "lobby", "selected_map", argv[2]);
			if (argc == 4) {
				if (string(argv[3]) == "--update_assets") {
					map_editor_update_assets = true;
				}
			}

			char orientation[2] = { '0' , '\0' };
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_orientation", orientation);
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_orientation_random_range", orientation);
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_orientation_random_range2", orientation);

			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_animationoffset", orientation);
			char scale[4] = {'1', '.', '0', '\0' };
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_scale", scale);
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_scale_random_range", scale);
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_scale_random_range2", scale);
			char zindex[4] = {'2', '5', '5', '\0' };
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_zindex", zindex);
			printf("map_to_edit: %s\n", argv[2]);
			mapeditor_init();
			ui_set_active("mapeditor_overlay");
			HANDLE game_thread_ = CreateThread(NULL, 0, mapeditor_thread, NULL, 0, NULL);
		}
	} else {
		HANDLE game_thread_ = CreateThread(NULL, 0, game_thread, NULL, 0, NULL);
	}

	while (running) {
		long tf_l = clock();

		while (SDL_PollEvent(&sdl_event) != 0) {
			if (game_started || map_editor) {
				if (sdl_event.type == SDL_KEYDOWN && sdl_event.key.keysym.sym == SDLK_ESCAPE) {
					if (map_editor) {
						ui_set_active("mapeditor_menu");
					} else {
						ui_set_active("ingame_menu");
					}
				}

				float camera_delta_z = 0.0f;
				if (sdl_event.type == SDL_MOUSEWHEEL) {
					if (!ui_process_scroll(&bf_rw, mouse_position[0], mouse_position[1], sdl_event.wheel.y)) {
						camera_delta_z -= sdl_event.wheel.y * sensitivity_z;
						camera_move(struct vector3<float>(0.0f, 0.0f, camera_delta_z));
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
					bool processed_click = ui_process_click(&bf_rw, mouse_position[0], mouse_position[1]);
					if (processed_click) {
						if (uis[ui_active].active_element_id > -1 && uis[ui_active].ui_elements[uis[ui_active].active_element_id].uet == UET_SCROLLLIST) {
							string ui_element_name = uis[ui_active].ui_elements[uis[ui_active].active_element_id].name;
							if (ui_active == "lobby" && ui_element_name == "maps") {
								if (uis[ui_active].active_element_param > -1) {
									string map_name = map_name_from_index(&bf_rw, uis[ui_active].active_element_param);
									ui_textfield_set_value(&bf_rw, "lobby", "selected_map", map_name.c_str());
									string map_asset_path = "./maps/" + map_name + "_minimap.png";
									ui_value_as_config(&bf_rw, "lobby", "minimap", 0, assets[map_asset_path]);
									ui_value_as_config(&bf_rw, "lobby", "minimap", 1, assets_dimensions[map_asset_path].width);
									ui_value_as_config(&bf_rw, "lobby", "minimap", 2, assets_dimensions[map_asset_path].height);
								}
							}
						}
					} else {
						if (map_editor && ui_active == "mapeditor_overlay") {
							printf("placing object\n");
							WaitForSingleObject(bf_rw.device_locks[0], INFINITE);
							mapeditor_place_object();
							ReleaseMutex(bf_rw.device_locks[0]);
						}
					}
				}
			}
		}

		WaitForSingleObject(bf_assets.device_locks[0], INFINITE);
		WaitForSingleObject(bf_rw.device_locks[0], INFINITE);
		bit_field_update_device(&bf_rw, 0);
		if (game_started || map_editor) {
			camera_move_z_tick();
			camera_get_crop(camera_crop);
			launch_draw_map(bf_map.device_data[0], gm.map_zoom_level_count, gm.map_zoom_center_z, gm.map_zoom_level_offsets_position, gm.map_positions, resolution[0], resolution[1], 4, camera_crop[0], camera_crop[1], camera_crop[2], camera_crop[3], bf_output.device_data[0], output_position, 1920, 1080);
			launch_draw_entities_kernel(bf_assets.device_data[0], bf_map.device_data[0], player_models_position, item_models_position, map_models_position, ui_fonts_position, bf_rw.device_data[0], entities_position, gd.position_in_bf, gd.data_position_in_bf,
				bf_output.device_data[0], output_position, 1920, 1080, 4, camera_crop[0], camera_crop[2], camera[2], mouse_position, game_ticks);
			if (!map_editor) {
				launch_draw_storm_kernel(bf_output.device_data[0], output_position, resolution[0], resolution[1], 4, camera_crop[0], camera_crop[2], camera[2], storm_current, storm_to, 50, { 45, 0, 100 });
			}
		}

		if (ui_active != "") {
			launch_draw_ui_kernel(bf_assets.device_data[0], uis[ui_active].background_position, ui_fonts_position, bf_output.device_data[0], output_position, resolution[0], resolution[1], 4, bf_rw.device_data[0], ui_tick_counter);
		}
		ReleaseMutex(bf_rw.device_locks[0]);
		ReleaseMutex(bf_assets.device_locks[0]);

		bit_field_update_host(&bf_output, 0);

		sdl_update_frame((Uint32*)&bf_output.data[output_position]);

		if (ui_tick_counter % 30 == 0) {
			if (!game_started) {
				if (ui_active == "loading_game" && !game_setup) {
					game_setup = true;
				}
				if (ui_active == "lobby" && irc_started) {
					twitch_update_players(&bf_rw);
					ui_textfield_set_int(&bf_rw, "lobby", "playercount", players.size());
				}
				if (ui_active == "lobby" && !irc_started) {
					string twitch_name = ui_textfield_get_value(&bf_rw, "settings", "twitch_name");
					bits_per_bandage = stoi(ui_textfield_get_value(&bf_rw, "settings", "bits_bandage"));
					bits_per_shield = stoi(ui_textfield_get_value(&bf_rw, "settings", "bits_shield"));
					max_bits_per_game = stoi(ui_textfield_get_value(&bf_rw, "settings", "bits_game"));
					players_max = stoi(ui_textfield_get_value(&bf_rw, "settings", "max_players"));

					if (twitch_name != "") {
						playerlist_init(&bf_rw);

						string map_name = map_name_from_index(&bf_rw, 0);
						ui_textfield_set_value(&bf_rw, "lobby", "selected_map", map_name.c_str());
						string map_asset_path = "./maps/" + map_name + "_minimap.png";
						ui_value_as_config(&bf_rw, "lobby", "minimap", 0, assets[map_asset_path]);
						ui_value_as_config(&bf_rw, "lobby", "minimap", 1, assets_dimensions[map_asset_path].width);
						ui_value_as_config(&bf_rw, "lobby", "minimap", 2, assets_dimensions[map_asset_path].height);

						twitch_launch_irc(twitch_name);
						irc_started = true;
					}
				}
			}
		}

		long tf_3 = clock();
		double frame_time = (tf_3 - tf_l) / (double)CLOCKS_PER_SEC;
		double frame_time_target = 1000.0f / (float)ui_ticks_per_second / 1000.0f;
		if (frame_time < frame_time_target) {
			frame_balancing = (int)floorf((frame_time_target - frame_time)*1000.0f);
		} else {
			frame_balancing = 0;
		}
			
		if (frame_balancing > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(frame_balancing));
		}
		tf = clock();
		ui_tick_counter++;
	}

	string s_fields[5] = { "twitch_name", "bits_bandage", "bits_shield", "bits_game", "max_players" };
	ui_save_fields_to_file(&bf_rw, "settings", s_fields, 5, "./", "settings.cfg");

	return 0;
}