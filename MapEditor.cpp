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

unsigned int			mapeditor_selectedasset_id = UINT_MAX;
unsigned int			mapeditor_selectedasset_entity_id = UINT_MAX;

struct vector3<float>	mapeditor_selectedasset_position = { 0.0f, 0.0f, 0.0f};

DWORD WINAPI mapeditor_thread(LPVOID param) {
	unsigned int frame_balancing = 0;
	while (running) {
		WaitForSingleObject(bf_rw.device_locks[0], INFINITE);

		if (ui_active == "mapeditor_menu") {
			if (mapeditor_selectedasset_entity_id != UINT_MAX) {
				string scale = ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale");
				float scale_f = stof(scale);
				grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, mapeditor_selectedasset_position, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models[mapeditor_selectedasset_id - 100]) * map_models[mapeditor_selectedasset_id - 100].model_scale, mapeditor_selectedasset_entity_id);
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
			if (mapeditor_selectedasset_entity_id != UINT_MAX) {
				float current_mouse_game_x = camera_crop[0] + mouse_position[0] * camera[2];
				float current_mouse_game_y = camera_crop[2] + mouse_position[1] * camera[2];
				
				string scale = ui_textfield_get_value(&bf_rw, "mapeditor_menu", "asset_scale");
				float scale_f = stof(scale);

				grid_object_remove(&bf_rw, bf_rw.data, gd.position_in_bf, mapeditor_selectedasset_position, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models[mapeditor_selectedasset_id - 100])* map_models[mapeditor_selectedasset_id - 100].model_scale, mapeditor_selectedasset_entity_id);
				grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, { current_mouse_game_x, current_mouse_game_y, 0.0f}, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models[mapeditor_selectedasset_id - 100]) * map_models[mapeditor_selectedasset_id - 100].model_scale, mapeditor_selectedasset_entity_id);
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
				entity_add("static_asset", ET_STATIC_ASSET, mapeditor_selectedasset_id, 255);
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
		}

		ReleaseMutex(bf_rw.device_locks[0]);
		std::this_thread::sleep_for(std::chrono::milliseconds(16));

		game_ticks++;
	}
	return NULL;
}

void mapeditor_init() {
	assetlist_init(&bf_rw);

	//load first map
	map_load(&bf_assets, ui_textfield_get_value(&bf_rw, "lobby", "selected_map"));

	//entity grid
	grid_init(&bf_rw, &gd, struct vector3<float>((float)gm.map_dimensions[0], (float)gm.map_dimensions[1], 1.0f), struct vector3<float>(32.0f, 32.0f, 1.0f), struct vector3<float>(0, 0, 0));

	map_add_static_assets(&bf_assets, &bf_rw, &gd);

	camera = { 0.0f, 0.0f, 1.0f };

	printf("uploading entities\n");
	entities_upload(&bf_rw);

	printf("updating assets players\n");
	bit_field_update_device(&bf_assets, 0);
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
		c_ss << mapeditor_selectedasset_position[0] << ", " << mapeditor_selectedasset_position[1] << " : " << orientation << " : " << scale_f << " : 255 : " << animationoffset;
		cur_object.second = c_ss.str();
		map_static_assets.emplace_back(cur_object);
		entity_add("static_asset", ET_STATIC_ASSET, mapeditor_selectedasset_id, 255);
		struct entity* cur_e = &entities[entities.size() - 1];
		cur_e->position = mapeditor_selectedasset_position;
		cur_e->orientation = orientation;
		cur_e->scale = scale_f;
		cur_e->model_animation_offset = animationoffset;
		printf("adding static_asset_id %i, entity_id %i\n", mapeditor_selectedasset_id, entities.size() - 1);
		//FIXME: memset causing issue
		//bit_field_remove_bulk_from_segment(&bf_rw, entities_position - 1);
		entities_upload(&bf_rw);
		grid_object_add(&bf_rw, bf_rw.data, gd.position_in_bf, mapeditor_selectedasset_position, { scale_f, scale_f, scale_f }, { 0.0f, 0.0f, 0.0f }, model_get_max_position(&map_models[mapeditor_selectedasset_id - 100]) * map_models[mapeditor_selectedasset_id - 100].model_scale, entities.size() - 1);
	}
}

void mapeditor_save() {
	string filepath = "./maps/" + ui_textfield_get_value(&bf_rw, "lobby", "selected_map") + "/" + ui_textfield_get_value(&bf_rw, "lobby", "selected_map") + "_static_assets.cfg";

	ofstream sfile;
	sfile.open(filepath);
	for (int i = 0; i < map_static_assets.size(); i++) {
		sfile << map_static_assets[i].first << " : " << map_static_assets[i].second << std::endl;
	}
	sfile.close();
}