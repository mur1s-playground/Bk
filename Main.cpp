#include "Main.hpp"

#include <sstream>

#include "BitField.hpp"
#include "AssetLoader.hpp"
#include "Map.hpp"
#include "Grid.hpp"
#include "Camera.hpp"
#include "SDLShow.hpp"
#include "Storm.hpp"
#include "Entity.hpp"

#include "time.h"
#include "math.h"

#include <ctime>
#include <random>


using namespace std;

struct bit_field bf_assets;
struct bit_field bf_rw;
struct bit_field bf_output;

struct vector3<float> camera;
struct vector2<unsigned int> resolution;

struct vector2<unsigned int> map_dimensions;

unsigned int tick_counter = 0;

int main(int argc, char** argv) {
	resolution = {1920, 1080};
	unsigned int output_size = resolution[0] * resolution[1] * 4;
	unsigned int output_size_in_bf = (int)ceilf(output_size / (float) sizeof(unsigned int));
	unsigned int output_position;

	srand(time(nullptr));

	/////////////////
	// -- ASSETS --//
	/////////////////

	//bit field uploaded once
	bit_field_init(&bf_assets, 16, 1024);
	bit_field_register_device(&bf_assets, 0);

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

	//uploaded & downloaded per frame
	bit_field_init(&bf_rw, 2048, 1024);
	bit_field_register_device(&bf_rw, 0);

	struct grid gd;
	grid_init(&bf_rw, &gd, struct vector3<float>((float)gm.map_dimensions[0], (float)gm.map_dimensions[1], 1.0f), struct vector3<float>(32.0f, 32.0f, 1.0f), struct vector3<float>(0, 0, 0));

	map_add_static_assets(&bf_assets, &bf_rw, &gd);
	
	//spawn players
	for (int i = 0; i < 512; i++) {
		stringstream ss_p;
		ss_p << i;

		entity_add("mur1_" + ss_p.str(), ET_PLAYER, 0, 0);
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
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, cur_e->position, { player_models[PT_HOPE].model_scale, player_models[PT_HOPE].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&player_models[PT_HOPE]), entities.size() - 1);

		player_add("mur1_" + ss_p.str(), PT_HOPE, entities.size()-1);
	}

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
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, cur_e->position, { item_models[0].model_scale, item_models[0].model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&item_models[0]), entities.size()-1);
	}

	entities_upload(&bf_rw);

	bit_field_update_device(&bf_rw, 0);

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

	bool running = true;
	while (running) {
		long tf = clock();
		long tf_l = clock();

		while (SDL_PollEvent(&sdl_event) != 0) {
			if (sdl_event.type == SDL_KEYDOWN && sdl_event.key.keysym.sym == SDLK_ESCAPE) {
				running = false;
			}

			float camera_delta_z = 0.0f;
			if (sdl_event.type == SDL_MOUSEWHEEL) {
				camera_delta_z -= sdl_event.wheel.y*sensitivity_z;
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
			if (sdl_event.type == SDL_MOUSEMOTION) {
				mouse_position[0] = sdl_event.motion.x;
				mouse_position[1] = sdl_event.motion.y;
			}

			if (sdl_event.type == SDL_MOUSEMOTION && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_RIGHT)) {
				float zoom_sensitivity = sensitivity_xy * camera[2] * sensitivity_zoom_ratio;
				if (zoom_sensitivity < 0.2f) zoom_sensitivity = 0.2f;
				camera_move(struct vector3<float>(-sdl_event.motion.xrel*zoom_sensitivity, -sdl_event.motion.yrel*zoom_sensitivity, 0.0f));
				camera_get_crop(camera_crop);
			}
		}

		struct entity* es = (struct entity*) &bf_rw.data[entities_position];
		for (int e = 0; e < entities.size(); e++) {
			struct entity* en = &es[e];
			if (en->et == ET_ITEM) {
				en->orientation += 1;
			}
		}
		bit_field_invalidate_bulk(&bf_rw, entities_position, entities_size_in_bf);
		bit_field_update_device(&bf_rw, 0);

		launch_draw_map(bf_assets.device_data[0], gm.map_zoom_level_count, gm.map_zoom_center_z, gm.map_zoom_level_offsets_position, gm.map_positions, resolution[0], resolution[1], 4, camera_crop[0], camera_crop[1], camera_crop[2], camera_crop[3], bf_output.device_data[0], output_position, 1920, 1080);

		launch_draw_entities_kernel(bf_assets.device_data[0], player_models_position, item_models_position, map_models_position, bf_rw.device_data[0], entities_position, gd.position_in_bf, gd.data_position_in_bf,
			bf_output.device_data[0], output_position, 1920, 1080, 4, camera_crop[0], camera_crop[2], camera[2], tick_counter);

		launch_draw_storm_kernel(bf_output.device_data[0], output_position, resolution[0], resolution[1], 4, camera_crop[0], camera_crop[2], camera[2], storm_current, storm_to, 50, { 45, 0, 100 });

		bit_field_update_host(&bf_output, 0);

		sdl_update_frame((Uint32*)&bf_output.data[output_position]);

		long tf_3 = clock();
		long tf_ = tf_3 - tf;
		sec += ((double)tf_ / CLOCKS_PER_SEC);
		fps++;

		if (sec >= 1.0) {
			printf("main fps: %d\r\n", fps);
			printf("main ft: %f, main ft_l: %f\r\n", tf_ / (double)CLOCKS_PER_SEC, (tf_3 - tf_l) / (double)CLOCKS_PER_SEC);
			sec = 0;
			fps = 0;
		}

		tick_counter++;
		storm_next(&bf_assets, &bf_rw);
	}
	return 0;
}