#ifndef TWITCHINTEGRATION_HPP
#define TWITCHINTEGRATION_HPP

#include <string>
#include "Main.hpp"
#include "BitField.hpp"

using namespace std;

void twitch_launch_irc(string twitch_name);
void twitch_terminate_irc();
void twitch_update_players(struct bit_field *bf_rw);
void twitch_update_bits();

#endif