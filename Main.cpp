#include "Main.hpp"

#include <vector>
#include <sstream>

#include "BitField.hpp"
#include "AssetLoader.hpp"
#include "Grid.hpp"
#include "Camera.hpp"
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

	map_dimensions = { 4096, 4096 };

	bit_field_init(&bf_rw, 2048, 1024);
	bit_field_register_device(&bf_rw, 0);

	struct grid gd;
	grid_init(&bf_rw, &gd, struct vector3<float> ((float)map_dimensions[0], (float)map_dimensions[1], 1.0f), struct vector3<float> (32.0f, 32.0f, 1.0f), struct vector3<float> (0, 0, 0));

	//player models
	vector<struct player_model> player_models;

	//hope
	vector<unsigned int> hope_model_positions;
	vector<unsigned int> hope_model_med_positions;
	vector<unsigned int> hope_model_lo_positions;
	vector<unsigned int> hope_shadow_positions;
	vector<unsigned int> hope_shadow_med_positions;
	vector<unsigned int> hope_shadow_lo_positions;

	struct player_model pm_hope;
	for (int i = 1; i <= 36; i++) {
		stringstream ls;
		if (i < 10) ls << 0;
		ls << i;
		hope_model_positions.push_back(assets["hope_model_hi__00" + ls.str() + ".png"]);
		hope_model_med_positions.push_back(assets["hope_model_med__00" + ls.str() + ".png"]);
		hope_model_lo_positions.push_back(assets["hope_model_lo__00" + ls.str() + ".png"]);

		hope_shadow_positions.push_back(assets["hope_model_shadow_hi__00" + ls.str() + ".png"]);
		hope_shadow_med_positions.push_back(assets["hope_model_shadow_med__00" + ls.str() + ".png"]);
		hope_shadow_lo_positions.push_back(assets["hope_model_shadow_lo__00" + ls.str() + ".png"]);
	}

	pm_hope.id = 0;
	pm_hope.model_scale = 1 / 8.0f;
	pm_hope.model_dimensions = { 256, 256, 1 };
	pm_hope.model_positions = bit_field_add_bulk(&bf_assets, hope_model_positions.data(), 36, 36*sizeof(unsigned int))+1;
	pm_hope.model_med_positions = bit_field_add_bulk(&bf_assets, hope_model_med_positions.data(), 36, 36 * sizeof(unsigned int)) + 1;
	pm_hope.model_lo_positions = bit_field_add_bulk(&bf_assets, hope_model_lo_positions.data(), 36, 36 * sizeof(unsigned int)) + 1;
	pm_hope.shadow_scale = 1 / 8.0f;
	pm_hope.shadow_dimensions = { 256, 256, 1 };
	pm_hope.shadow_offset = { 2, 9 };
	pm_hope.shadow_positions = bit_field_add_bulk(&bf_assets, hope_shadow_positions.data(), 36, 36*sizeof(unsigned int))+1;
	pm_hope.shadow_med_positions = bit_field_add_bulk(&bf_assets, hope_shadow_med_positions.data(), 36, 36 * sizeof(unsigned int)) + 1;
	pm_hope.shadow_lo_positions = bit_field_add_bulk(&bf_assets, hope_shadow_lo_positions.data(), 36, 36 * sizeof(unsigned int)) + 1;
	player_models.push_back(pm_hope);

	unsigned int player_model_info_size_in_bf = (unsigned int)(player_models.size() * ceilf(sizeof(struct player_model) / (float) sizeof(unsigned int)));
	unsigned int player_model_info_position = bit_field_add_bulk_zero(&bf_assets, player_model_info_size_in_bf)+1;
	struct player_model* player_models_ptr = (struct player_model*) &bf_assets.data[player_model_info_position];
	memcpy(player_models_ptr, player_models.data(), player_models.size() * sizeof(struct player_model));

	bit_field_update_device(&bf_assets, 0);

	//output
	bit_field_init(&bf_output, 16, 1024);
	output_position = bit_field_add_bulk_zero(&bf_output, output_size_in_bf)+1;
	bit_field_register_device(&bf_output, 0);

	srand(time(nullptr));
	vector<struct player> v_pl;
	struct vector3<float> position_max;
	for (int i = 0; i < 512; i++) {
		stringstream ss_p;
		ss_p << i;
		player_add("mur1_" + ss_p.str(), PT_HOPE);
		players["mur1_" + ss_p.str()].position = { 10.0f + rand() % (map_dimensions[0] - 32), 10.0f + rand() % (map_dimensions[1] - 32), 0.0f };
		players["mur1_" + ss_p.str()].model_id = 0;
		
		position_max[0] = max((float)((pm_hope.shadow_offset[0] + pm_hope.shadow_dimensions[0])), (float)(pm_hope.model_dimensions[0]));
		position_max[1] = max((float)((pm_hope.shadow_offset[1] + pm_hope.shadow_dimensions[1])), (float)(pm_hope.model_dimensions[1]));
		position_max[2] = 0.0f;
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, players["mur1_" + ss_p.str()].position, { pm_hope.model_scale, pm_hope.model_scale, 1.0f }, {0.0f, 0.0f, 0.0f}, position_max, i);
		int grid_index = grid_get_index(bf_rw.data, gd.position_in_bf, players["mur1_" + ss_p.str()].position);
		int data_pos = bf_rw.data[gd.data_position_in_bf + 1 + grid_index];
		int count = bf_rw.data[data_pos];
		v_pl.emplace_back(players["mur1_" + ss_p.str()]);
	}
	unsigned int players_size_in_bf = players.size() * (int)ceilf(sizeof(struct player) / (float)sizeof(unsigned int));
	unsigned int players_position = bit_field_add_bulk_zero(&bf_rw, players_size_in_bf)+1;
	struct player* players_ptr = (struct player *) &bf_rw.data[players_position];
	memcpy(players_ptr, v_pl.data(), players.size() * sizeof(struct player));
	
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
			grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, players_ptr[p].position, { pm_hope.model_scale, pm_hope.model_scale, 1.0f }, {0.0f, 0.0f, 0.0f}, position_max, p);
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
				position_max[0] = max((float)((pm_hope.shadow_offset[0] + pm_hope.shadow_dimensions[0])), (float)(pm_hope.model_dimensions[0]));
				position_max[1] = max((float)((pm_hope.shadow_offset[1] + pm_hope.shadow_dimensions[1])), (float)(pm_hope.model_dimensions[1]));
				position_max[2] = 0.0f;
				grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, players_ptr[p].position, { pm_hope.model_scale, pm_hope.model_scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, position_max, p);
				players_ptr = (struct player*) &bf_rw.data[players_position];
		}
		
		bit_field_invalidate_bulk(&bf_rw, players_position, players_size_in_bf);
		bit_field_update_device(&bf_rw, 0);

		launch_resize(bf_assets.device_data[0], assets["map_0.png"], map_dimensions[0], map_dimensions[1], 4, camera_crop[0], camera_crop[1], camera_crop[2], camera_crop[3], bf_output.device_data[0], output_position, 1920, 1080);

		launch_draw_players_kernel(bf_assets.device_data[0], player_model_info_position,
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