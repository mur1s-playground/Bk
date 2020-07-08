#include "MapEditor.hpp"

#include <thread>
#include <chrono>
#include <sstream>
#include <fstream>
#include <iostream>

#include "Main.hpp"
#include "Map.hpp"
#include "Grid.hpp"
#include "UI.hpp"
#include "Camera.hpp"
#include "Entity.hpp"
#include "AssetList.hpp"
#include "Vector3.hpp"
#include "Game.hpp"

#include "lodepng.h"

unsigned int			mapeditor_selectedasset_id = UINT_MAX;
unsigned int			mapeditor_selectedasset_entity_id = UINT_MAX;

struct vector3<float>	mapeditor_selectedasset_position = { 0.0f, 0.0f, 0.0f};

struct vector2<unsigned int> mapeditor_pathing_brushsize = { 1, 1 };
unsigned int			mapeditor_pathing_clear = 0;

unsigned int			mapeditor_action_type = 0;

DWORD WINAPI mapeditor_thread(LPVOID param) {
	unsigned int frame_balancing = 0;
	while (running) {
		WaitForSingleObject(bf_rw.device_locks[0], INFINITE);

		if (ui_active == "mapeditor_menu") {
			if (ui_value_get_int(&bf_rw, "mapeditor_menu", "asset_at_checkbox", 0) == 1) {
				mapeditor_action_type = 0;
			} else if (ui_value_get_int(&bf_rw, "mapeditor_menu", "pathing_at_checkbox", 0) == 1) {
				if (mapeditor_selectedasset_entity_id < UINT_MAX) {
					string scale = ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale");
					float scale_f = stof(scale);
					struct model* map_models_bf = (struct model*) &bf_map.data[map_models_position];
					grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, mapeditor_selectedasset_position, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models_bf[mapeditor_selectedasset_id - 100]) * map_models_bf[mapeditor_selectedasset_id - 100].model_scale, mapeditor_selectedasset_entity_id);
					mapeditor_selectedasset_id = UINT_MAX;
					mapeditor_selectedasset_entity_id = UINT_MAX;
				}
				mapeditor_action_type = 1;
			}
			if (mapeditor_selectedasset_entity_id != UINT_MAX) {
				string scale = ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale");
				float scale_f = stof(scale);
				struct model* map_models_bf = (struct model*) & bf_map.data[map_models_position];
				grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, mapeditor_selectedasset_position, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models_bf[mapeditor_selectedasset_id - 100]) * map_models_bf[mapeditor_selectedasset_id - 100].model_scale, mapeditor_selectedasset_entity_id);
				mapeditor_selectedasset_entity_id = UINT_MAX;
			}
			if (uis["mapeditor_menu"].active_element_id > -1 &&
				(uis["mapeditor_menu"].ui_elements[uis["mapeditor_menu"].active_element_id].name == "assetlist_id" || uis["mapeditor_menu"].ui_elements[uis["mapeditor_menu"].active_element_id].name == "assetlist_name")) {
				//assetlist_id_element
				struct assetlist_id_element* lp = (struct assetlist_id_element*) &bf_rw.data[assetlist_id_pos];
				string asset_id = string(lp[uis["mapeditor_menu"].active_element_param].value);
				if (mapeditor_selectedasset_id != stoi(asset_id)) {
					mapeditor_selectedasset_entity_id = UINT_MAX;
					mapeditor_selectedasset_id = stoi(asset_id);
				}
			}
		}

		if (ui_active == "mapeditor_overlay") {
			if (mapeditor_action_type == 0) {
				if (mapeditor_selectedasset_entity_id != UINT_MAX) {
					float current_mouse_game_x = camera_crop[0] + mouse_position[0] * camera[2];
					float current_mouse_game_y = camera_crop[2] + mouse_position[1] * camera[2];

					string scale = ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale");
					float scale_f = stof(scale);

					struct model* map_models_bf = (struct model*) & bf_map.data[map_models_position];
					grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, mapeditor_selectedasset_position, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models_bf[mapeditor_selectedasset_id - 100]) * map_models_bf[mapeditor_selectedasset_id - 100].model_scale, mapeditor_selectedasset_entity_id);
					grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { current_mouse_game_x, current_mouse_game_y, 0.0f }, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models_bf[mapeditor_selectedasset_id - 100]) * map_models_bf[mapeditor_selectedasset_id - 100].model_scale, mapeditor_selectedasset_entity_id);
					mapeditor_selectedasset_position = { current_mouse_game_x, current_mouse_game_y, 0.0f };

					struct entity* es = (struct entity*) & bf_rw.data[entities_position];
					struct entity* en = (struct entity*) & es[mapeditor_selectedasset_entity_id];
					en->position = { current_mouse_game_x, current_mouse_game_y, 0.0f };
					string orientation = ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_orientation");
					en->orientation = stoi(orientation);
					en->scale = scale_f;
					string animationoffset = ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_animationoffset");
					en->model_animation_offset = stoi(animationoffset);

					bit_field_invalidate_bulk(&bf_rw, entities_position, entities_size_in_bf);

					struct entity* cur_e = &entities[mapeditor_selectedasset_entity_id];
					cur_e->position = { current_mouse_game_x, current_mouse_game_y, 0.0f };
					cur_e->orientation = stoi(orientation);
					cur_e->scale = scale_f;
					cur_e->model_animation_offset = stoi(animationoffset);
				}
				if (mapeditor_selectedasset_id < UINT_MAX && mapeditor_selectedasset_entity_id == UINT_MAX) {
					entity_add("static_asset", ET_STATIC_ASSET, mapeditor_selectedasset_id, stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_zindex")));
					mapeditor_selectedasset_entity_id = entities.size() - 1;
					struct entity* cur_e = &entities[entities.size() - 1];
					cur_e->position = { 0.0f, 0.0f, 0.0f };
					cur_e->orientation = stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_orientation"));
					cur_e->scale = stof(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale"));
					cur_e->model_animation_offset = stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_animationoffset"));
					printf("adding static_asset_id %i, entity_id %i\n", mapeditor_selectedasset_id, mapeditor_selectedasset_entity_id);
					//FIXME: memset causing issue
					//bit_field_remove_bulk_from_segment(&bf_rw, entities_position - 1);
					entities_upload(&bf_rw);
				}
			} else if (mapeditor_action_type == 1) {
				mapeditor_pathing_brushsize = { (unsigned int)stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "pathing_brushsize_x")), (unsigned int)stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "pathing_brushsize_y")) };
				mapeditor_pathing_clear = ui_value_get_int(&bf_rw, "mapeditor_menu", "pathing_clear_checkbox", 0);
			}
		}

		ReleaseMutex(bf_rw.device_locks[0]);
		std::this_thread::sleep_for(std::chrono::milliseconds(32));

		game_ticks++;
	}
	return NULL;
}

void mapeditor_init() {
	assetlist_init(&bf_rw);

	//load first map
	map_load(&bf_map, ui_textfield_get_value(&bf_rw, "lobby", "selected_map"));

	//entity grid
	grid_init(&bf_rw, &gd, struct vector3<float>((float)gm.map_dimensions[0], (float)gm.map_dimensions[1], 1.0f), struct vector3<float>(32.0f, 32.0f, 1.0f), struct vector3<float>(0, 0, 0));

	map_add_static_assets(&bf_map, &bf_rw, &gd);

	camera = { 0.0f, 0.0f, 1.0f };

	printf("uploading entities\n");
	entities_upload(&bf_rw);

	printf("updating assets players\n");
	bit_field_update_device(&bf_assets, 0);
	printf("updated assets players\n");
}

void mapeditor_process_click() {
	WaitForSingleObject(bf_rw.device_locks[0], INFINITE);
	if (mapeditor_action_type == 0) {
		printf("placing object\n");
		mapeditor_place_object();
	} else if (mapeditor_action_type == 1){
		mapeditor_draw_pathing();
	}
	ReleaseMutex(bf_rw.device_locks[0]);
}

void mapeditor_place_object() {
	if (mapeditor_selectedasset_id < UINT_MAX && mapeditor_selectedasset_entity_id < UINT_MAX) {
		pair<string, string> cur_object;
		stringstream id_ss;
		id_ss << mapeditor_selectedasset_id;
		cur_object.first = id_ss.str();
		stringstream c_ss;
		unsigned int orientation = stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_orientation"));
		unsigned int animationoffset = stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_animationoffset"));
		float scale_f = stof(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale"));

		unsigned int zindex = stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_zindex"));

		c_ss << mapeditor_selectedasset_position[0] << ", " << mapeditor_selectedasset_position[1] << " : " << orientation << " : " << scale_f << " : " << zindex << " : " << animationoffset;
		cur_object.second = c_ss.str();
		map_static_assets.emplace_back(cur_object);
		entity_add("static_asset", ET_STATIC_ASSET, mapeditor_selectedasset_id, zindex);
		struct entity* cur_e = &entities[entities.size() - 1];
		cur_e->position = mapeditor_selectedasset_position;
		cur_e->orientation = orientation;
		cur_e->scale = scale_f;
		cur_e->model_animation_offset = animationoffset;
		printf("adding static_asset_id %i, entity_id %i\n", mapeditor_selectedasset_id, entities.size() - 1);
		//FIXME: memset causing issue
		//bit_field_remove_bulk_from_segment(&bf_rw, entities_position - 1);
		entities_upload(&bf_rw);
		struct model* map_models_bf = (struct model*) & bf_map.data[map_models_position];
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, mapeditor_selectedasset_position, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models_bf[mapeditor_selectedasset_id - 100]) * map_models_bf[mapeditor_selectedasset_id - 100].model_scale, entities.size() - 1);
		if (ui_value_get_int(&bf_rw, "mapeditor_menu", "asset_scale_random", 0) == 1) {
			float rand_val = stof(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale_random_range")) + (rand() / (float)RAND_MAX)*(stof(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale_random_range2"))-stof(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale_random_range")));
			stringstream r_ss;
			r_ss << rand_val;
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_scale", r_ss.str().c_str());
		}
		if (ui_value_get_int(&bf_rw, "mapeditor_menu", "asset_orientation_random", 0) == 1) {
			int rand_val = stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_orientation_random_range")) + (int)roundf((rand() / (float)RAND_MAX) * (stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_orientation_random_range2")) - stoi(ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_orientation_random_range"))));
			stringstream r_ss;
			r_ss << rand_val;
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_orientation", r_ss.str().c_str());
		}
		if (ui_value_get_int(&bf_rw, "mapeditor_menu", "asset_animationoffset_random", 0) == 1) {
			int rand_val = (int)roundf((rand() / (float)RAND_MAX) * (map_models_bf[mapeditor_selectedasset_id - 100].model_animation_ticks-1));
			stringstream r_ss;
			r_ss << rand_val;
			ui_textfield_set_value(&bf_rw, "mapeditor_menu", "asset_animationoffset", r_ss.str().c_str());
		}
	}
}

void mapeditor_draw_pathing() {
	int x = camera_crop[0]+mouse_position[0]*camera[2];
	int y = camera_crop[2]+mouse_position[1]*camera[2];
	unsigned char* pathables = (unsigned char *)&bf_map.data[gm.map_pathable_position];
	for (int i = 0; i < mapeditor_pathing_brushsize[1]; i++) {
		for (int j = 0; j < mapeditor_pathing_brushsize[0]; j++) {
			if (y + i < gm.map_dimensions[1] && x + j < gm.map_dimensions[0]) {
				if (mapeditor_pathing_clear == 1) {
					pathables[(y + i) * gm.map_dimensions[0] + (x + j)] = 255;
				} else {
					pathables[(y + i) * gm.map_dimensions[0] + (x + j)] = 0;
				}
			}
		}
	}
	int size = gm.map_dimensions[0] * gm.map_dimensions[1];
	int size_in_bf = ceil(size / (float)sizeof(unsigned int));
	bit_field_invalidate_bulk(&bf_map, gm.map_pathable_position, size_in_bf);
	bit_field_update_device(&bf_map, 0);
}

void mapeditor_save() {
	string name = ui_textfield_get_value(&bf_rw, "lobby", "selected_map");

	string filepath = "./maps/" + name + "/" + name + "_static_assets.cfg";

	ofstream sfile;
	sfile.open(filepath);
	for (int i = 0; i < map_static_assets.size(); i++) {
		sfile << map_static_assets[i].first << " : " << map_static_assets[i].second << std::endl;
	}
	sfile.close();

	vector<unsigned char> pathing_in;
	pathing_in.reserve(gm.map_dimensions[0]*gm.map_dimensions[1]);
	unsigned char* pathables = (unsigned char*)&bf_map.data[gm.map_pathable_position];
	for (int y = 0; y < gm.map_dimensions[1]; y++) {
		for (int x = 0; x < gm.map_dimensions[0]; x++) {
			pathing_in.push_back(pathables[y*gm.map_dimensions[0]+x]);
		}
	}
	vector<unsigned char> pathing_out;
	lodepng::encode(pathing_out, pathing_in, gm.map_dimensions[0], gm.map_dimensions[1], LCT_GREY);
	lodepng::save_file(pathing_out, "./maps/" + name + "/" + name + ".pathable.png");

	if (map_editor_update_assets) {
		bit_field_save_to_disk(&bf_map, "./maps/" + name + "/" + name + ".bf");

		string filepath = "./maps/" + name + "/" + name + ".bf.txt";

		ofstream bfile;
		bfile.open(filepath);

		bfile << "map_zoom_level_offsets_position	:	" << gm.map_zoom_level_offsets_position << std::endl;
		bfile << "map_loot_probabilities_position	:	" << gm.map_loot_probabilities_position << std::endl;
		bfile << "map_pathable_position				:	" << gm.map_pathable_position << std::endl;
		bfile << "map_positions						:	" << gm.map_positions << std::endl;
		bfile << "map_spawn_probabilities_position	:	" << gm.map_spawn_probabilities_position << std::endl;
		bfile << "map_models_position				:	" << map_models_position << std::endl;

		bfile.close();
	}
}