#ifndef TWITCHINTEGRATION_HPP
#define TWITCHINTEGRATION_HPP

#include <string>

using namespace std;

void twitch_launch_irc(string cache_dir_, string twitch_name);
void twitch_terminate_irc();
void twitch_update_players();

#endif