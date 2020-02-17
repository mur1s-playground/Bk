#include "Main.hpp"

#include <vector>
#include <sstream>

#include "BitField.hpp"
#include "AssetLoader.hpp"
#include "Grid.hpp"
#include "Camera.hpp"
#include "Model.hpp"
#include "Player.hpp"
#include "SDLShow.hpp"
#include "Resize.hpp"
#include "Copy.hpp"
#include "DrawPlayers.hpp"
#include "Storm.hpp"

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

	//load all assets
	bit_field_init(&bf_assets, 16, 1024);
	bit_field_register_device(&bf_assets, 0);
	asset_loader_load_all(&bf_assets);

	map_dimensions = { 8192, 8192 };

	bit_field_init(&bf_rw, 2048, 1024);
	bit_field_register_device(&bf_rw, 0);

	struct grid gd;
	grid_init(&bf_rw, &gd, struct vector3<float> ((float)map_dimensions[0], (float)map_dimensions[1], 1.0f), struct vector3<float> (32.0f, 32.0f, 1.0f), struct vector3<float> (0, 0, 0));

	//models
	vector<struct model> models;

	//hope model
	vector<string> hope_file_prefixes;
	hope_file_prefixes.push_back("hope_model_hi__00");
	hope_file_prefixes.push_back("hope_model_med__00");
	hope_file_prefixes.push_back("hope_model_lo__00");
	hope_file_prefixes.push_back("hope_model_shadow_hi__00");
	hope_file_prefixes.push_back("hope_model_shadow_med__00");
	hope_file_prefixes.push_back("hope_model_shadow_lo__00");
	
	struct model m_hope;
	model_init(&m_hope, &bf_assets, hope_file_prefixes, 0, MT_PLAYER, 1, 36, 1 / 8.0f, struct vector3<unsigned int>(256, 256, 1), 1 / 8.0f, struct vector3<unsigned int>(256, 256, 1), struct vector2<unsigned int>(2, 9));
	models.push_back(m_hope);
	//

	//tree model
	vector<string> tree_file_prefixes;
	tree_file_prefixes.push_back("tree_model_hi__00");
	tree_file_prefixes.push_back("tree_model_med__00");
	tree_file_prefixes.push_back("tree_model_lo__00");
	tree_file_prefixes.push_back("tree_model_shadow_hi__00");
	tree_file_prefixes.push_back("tree_model_shadow_med__00");
	tree_file_prefixes.push_back("tree_model_shadow_lo__00");

	struct model m_tree;
	model_init(&m_tree, &bf_assets, tree_file_prefixes, 1, MT_STATIC_ASSET, 1, 36, 1 / 4.0f, struct vector3<unsigned int>(512, 512, 1), 1 / 4.0f, struct vector3<unsigned int>(512, 512, 1), struct vector2<unsigned int>(10, 20));
	models.push_back(m_tree);
	//
	
	//colt model
	vector<string> colt_file_prefixes;
	colt_file_prefixes.push_back("colt_model_hi__00");
	colt_file_prefixes.push_back("colt_model_med__00");
	colt_file_prefixes.push_back("colt_model_lo__00");
	colt_file_prefixes.push_back("colt_model_shadow_hi__00");
	colt_file_prefixes.push_back("colt_model_shadow_med__00");
	colt_file_prefixes.push_back("colt_model_shadow_lo__00");

	struct model m_colt;
	model_init(&m_colt, &bf_assets, colt_file_prefixes, 2, MT_LOOTABLE_ITEM, 1, 36, 1/ 16.0f, struct vector3<unsigned int>(256, 256, 1), 1 / 16.0f, struct vector3<unsigned int>(256, 256, 1), struct vector2<unsigned int>(3, 4));
	models.push_back(m_colt);
	//
	
	unsigned int models_size_in_bf = (unsigned int)(models.size() * ceilf(ceilf(sizeof(struct model) / (float)sizeof(unsigned int))));
	unsigned int models_position = bit_field_add_bulk_zero(&bf_assets, models_size_in_bf) + 1;
	struct model* models_ptr = (struct model*) & bf_assets.data[models_position];
	memcpy(models_ptr, models.data(), models.size() * sizeof(struct model));

	bit_field_update_device(&bf_assets, 0);

	//output
	bit_field_init(&bf_output, 16, 1024);
	output_position = bit_field_add_bulk_zero(&bf_output, output_size_in_bf)+1;
	bit_field_register_device(&bf_output, 0);

	srand(time(nullptr));
	vector<struct player> v_pl;
	for (int i = 0; i < 512; i++) {
		stringstream ss_p;
		ss_p << i;
		player_add("mur1_" + ss_p.str(), PT_HOPE, 0);
		players["mur1_" + ss_p.str()].position = { 10.0f + rand() % (map_dimensions[0] - 32), 10.0f + rand() % (map_dimensions[1] - 32), 0.0f };
		
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, players["mur1_" + ss_p.str()].position, { m_hope.model_scale, m_hope.model_scale, 1.0f }, {0.0f, 0.0f, 0.0f}, model_get_max_position(&m_hope), i);
		v_pl.emplace_back(players["mur1_" + ss_p.str()]);
	}
	for (int i = 0; i < 20; i++) {
		stringstream ss_p;
		ss_p << i;
		player_add("tree_" + ss_p.str(), PT_HOPE, 1);
		players["tree_" + ss_p.str()].position = { 200.0f + (i % 10)* 100, 200.0f + (i / 10) * 100, 0.0f };
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, players["tree_" + ss_p.str()].position, { m_tree.model_scale, m_tree.model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&m_tree), 512+i);
		v_pl.emplace_back(players["tree_" + ss_p.str()]);
	}
	for (int i = 0; i < 20; i++) {
		stringstream ss_p;
		ss_p << i;
		player_add("colt_" + ss_p.str(), PT_HOPE, 2);
		players["colt_" + ss_p.str()].position = { 450.0f + (i % 10) * 100, 450.0f + (i / 10) * 100, 0.0f };
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, players["colt_" + ss_p.str()].position, { m_colt.model_scale, m_colt.model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&m_colt), 512 + 20 + i);
		v_pl.emplace_back(players["colt_" + ss_p.str()]);
	}

	unsigned int players_size_in_bf = v_pl.size() * (int)ceilf(sizeof(struct player) / (float)sizeof(unsigned int));
	unsigned int players_position = bit_field_add_bulk_zero(&bf_rw, players_size_in_bf)+1;
	struct player* players_ptr = (struct player *) &bf_rw.data[players_position];
	memcpy(players_ptr, v_pl.data(), v_pl.size() * sizeof(struct player));
	
	bit_field_update_device(&bf_rw, 0);

	camera = { 0.0f, 0.0f, 1.0f };

	storm_init();
	
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

		for (int p = 0; p < 512; p++) {
			grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, players_ptr[p].position, { m_hope.model_scale, m_hope.model_scale, 1.0f }, {0.0f, 0.0f, 0.0f}, model_get_max_position(&m_hope), p);
		}
		for (int p = 0; p < 512; p++) {
				float rand_dir = (rand() / (float)RAND_MAX - 0.5f) * 2;
				if (players_ptr[p].position[0] + rand_dir < map_dimensions[0] - 32 && players_ptr[p].position[0] + rand_dir >= 10) {
					players_ptr[p].position[0] += rand_dir;
				}
				if (players_ptr[p].position[1] + rand_dir < map_dimensions[1] - 32 && players_ptr[p].position[1] + rand_dir >= 10) {
					players_ptr[p].position[1] += rand_dir;
				}
				
				players_ptr[p].orientation += (rand_dir - 0.5);
				if (players_ptr[p].orientation < 0) players_ptr[p].orientation += 360;
				//players_ptr[0].position[0] += 0.05f;
				grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, players_ptr[p].position, { m_hope.model_scale, m_hope.model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&m_hope), p);
				players_ptr = (struct player*) &bf_rw.data[players_position];
		}
		for (int p = 512 + 20; p < 512 + 20 + 20; p++) {
			players_ptr[p].orientation += 1;
		}
		
		bit_field_invalidate_bulk(&bf_rw, players_position, players_size_in_bf);
		bit_field_update_device(&bf_rw, 0);

		launch_resize(bf_assets.device_data[0], assets["map_0.png"], map_dimensions[0], map_dimensions[1], 4, camera_crop[0], camera_crop[1], camera_crop[2], camera_crop[3], bf_output.device_data[0], output_position, 1920, 1080);

		launch_draw_players_kernel(bf_assets.device_data[0], models_position,
			bf_rw.device_data[0], players_position, gd.position_in_bf, gd.data_position_in_bf,
			bf_output.device_data[0], output_position, resolution[0], resolution[1], 4,
			camera_crop[0], camera_crop[2], camera[2], tick_counter);

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
		storm_next();
	}
	
	return 0;
}