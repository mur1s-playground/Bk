#ifndef TWITCHINTEGRATION_HPP
#define TWITCHINTEGRATION_HPP

#include <string>
#include "Main.hpp"

using namespace std;

void twitch_launch_irc(string cache_dir_, string twitch_name);
void twitch_terminate_irc();
void twitch_update_players();
void twitch_update_bits();

#endif