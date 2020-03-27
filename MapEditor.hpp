#ifndef MAPEDITOR_HPP
#define MAPEDITOR_HPP

#include <windows.h>

#include "Vector2.hpp"

extern unsigned int mapeditor_selectedasset_id;
extern unsigned int mapeditor_action_type;
extern vector2<unsigned int>	mapeditor_pathing_brushsize;

DWORD WINAPI mapeditor_thread(LPVOID param);
void mapeditor_init();
void mapeditor_process_click();
void mapeditor_place_object();
void mapeditor_draw_pathing();
void mapeditor_save();

#endif