#ifndef GAME_HPP
#define GAME_HPP

#include <atomic>
#include <windows.h>

using namespace std;

extern atomic<unsigned int>	game_ticks;
extern atomic<unsigned int>	game_ticks_target;
extern atomic<bool> game_setup;
extern atomic<bool> game_started;
extern atomic<bool> game_tick_ready;

DWORD WINAPI game_thread(LPVOID param);
void game_init();
void game_start();
void game_tick();
void game_destroy();

#endif