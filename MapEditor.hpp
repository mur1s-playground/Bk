#ifndef MAPEDITOR_HPP
#define MAPEDITOR_HPP

#include <windows.h>

extern unsigned int mapeditor_selectedasset_id;

DWORD WINAPI mapeditor_thread(LPVOID param);
void mapeditor_init();
void mapeditor_place_object();
void mapeditor_save();

#endif