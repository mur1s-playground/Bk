/*
 * Copyright (C) 2011 Fredi Machado <https://github.com/fredimachado>
 * IRCClient is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * http://www.gnu.org/licenses/lgpl.html 
 */

#include "IRCHandler.h"

#include <vector>
#include <algorithm>

std::atomic<bool> player_reg_active = true;
std::string cache_dir;
std::vector<std::string> players;

IRCCommandHandler ircCommandTable[NUM_IRC_CMDS] =
{
    { "PRIVMSG",            &IRCClient::HandlePrivMsg                   },
    { "NOTICE",             &IRCClient::HandleNotice                    },
    { "JOIN",               &IRCClient::HandleChannelJoinPart           },
    { "PART",               &IRCClient::HandleChannelJoinPart           },
    { "NICK",               &IRCClient::HandleUserNickChange            },
    { "QUIT",               &IRCClient::HandleUserQuit                  },
    { "353",                &IRCClient::HandleChannelNamesList          },
    { "433",                &IRCClient::HandleNicknameInUse             },
    { "001",                &IRCClient::HandleServerMessage             },
    { "002",                &IRCClient::HandleServerMessage             },
    { "003",                &IRCClient::HandleServerMessage             },
    { "004",                &IRCClient::HandleServerMessage             },
    { "005",                &IRCClient::HandleServerMessage             },
    { "250",                &IRCClient::HandleServerMessage             },
    { "251",                &IRCClient::HandleServerMessage             },
    { "252",                &IRCClient::HandleServerMessage             },
    { "253",                &IRCClient::HandleServerMessage             },
    { "254",                &IRCClient::HandleServerMessage             },
    { "255",                &IRCClient::HandleServerMessage             },
    { "265",                &IRCClient::HandleServerMessage             },
    { "266",                &IRCClient::HandleServerMessage             },
    { "366",                &IRCClient::HandleServerMessage             },
    { "372",                &IRCClient::HandleServerMessage             },
    { "375",                &IRCClient::HandleServerMessage             },
    { "376",                &IRCClient::HandleServerMessage             },
    { "439",                &IRCClient::HandleServerMessage             },
};

void IRCClient::HandleCTCP(IRCMessage message)
{
    std::string to = message.parameters.at(0);
    std::string text = message.parameters.at(message.parameters.size() - 1);

    // Remove '\001' from start/end of the string
    text = text.substr(1, text.size() - 2);

    std::cout << "[" + message.prefix.nick << " requested CTCP " << text << "]" << std::endl;

    if (to == _nick)
    {
        if (text == "VERSION") // Respond to CTCP VERSION
        {
            //SendIRC("NOTICE " + message.prefix.nick + " :\001VERSION Open source IRC client by Fredi Machado - https://github.com/fredimachado/IRCClient \001");
            return;
        }

        // CTCP not implemented
        //SendIRC("NOTICE " + message.prefix.nick + " :\001ERRMSG " + text + " :Not implemented\001");
    }
}

void IRCClient::HandlePrivMsg(IRCMessage message)
{
    std::string to = message.parameters.at(0);
    std::string text = message.parameters.at(message.parameters.size() - 1);

    // Handle Client-To-Client Protocol
    if (text[0] == '\001')
    {
        HandleCTCP(message);
        return;
    }

    if (to[0] == '#') {
        std::cout << "From " + message.prefix.nick << " @ " + to + ": " << text << std::endl;
        if (player_reg_active) {
            if (text[0] == '!') {
                if (text == "!add" || text == "!play") {
                    if (std::count(players.begin(), players.end(), message.prefix.nick) == 0) {
                        byte buffer[4096];
                        DWORD dwBytesWritten, dwPos;
                        std::string filepath = cache_dir + "\\players.txt";
                        HANDLE hAppend = CreateFile(TEXT(filepath.c_str()),
                            FILE_APPEND_DATA,         // open for writing
                            FILE_SHARE_READ,          // allow multiple readers
                            NULL,                     // no security
                            OPEN_ALWAYS,              // open or create
                            FILE_ATTRIBUTE_NORMAL,    // normal file
                            NULL);                    // no attr. template
                        if (hAppend == INVALID_HANDLE_VALUE) {
                            printf("could not open players.txt");
                        }
                        else {
                            dwPos = SetFilePointer(hAppend, 0, NULL, FILE_END);
                            int nicklen_bytes = (message.prefix.nick.length() + 1);
                            std::string linetoadd = message.prefix.nick + "\n";
                            LockFile(hAppend, dwPos, 0, nicklen_bytes, 0);
                            WriteFile(hAppend, linetoadd.c_str(), nicklen_bytes, &dwBytesWritten, NULL);
                            UnlockFile(hAppend, dwPos, 0, nicklen_bytes, 0);
                            CloseHandle(hAppend);
                            players.push_back(message.prefix.nick);
                        }
                    }
                }
            }
        }
    } else {
        std::cout << "From " + message.prefix.nick << ": " << text << std::endl;
    }
}

void IRCClient::HandleNotice(IRCMessage message)
{
    std::string from = message.prefix.nick != "" ? message.prefix.nick : message.prefix.prefix;
    std::string text;

    if( !message.parameters.empty() )
        text = message.parameters.at(message.parameters.size() - 1);

    if (!text.empty() && text[0] == '\001')
    {
        text = text.substr(1, text.size() - 2);
        if (text.find(" ") == std::string::npos)
        {
            std::cout << "[Invalid " << text << " reply from " << from << "]" << std::endl;
            return;
        }
        std::string ctcp = text.substr(0, text.find(" "));
        std::cout << "[" << from << " " << ctcp << " reply]: " << text.substr(text.find(" ") + 1) << std::endl;
    }
    else
        std::cout << "-" << from << "- " << text << std::endl;
}

void IRCClient::HandleChannelJoinPart(IRCMessage message)
{
    std::string channel = message.parameters.at(0);
    std::string action = message.command == "JOIN" ? "joins" : "leaves";
    std::cout << message.prefix.nick << " " << action << " " << channel << std::endl;
}

void IRCClient::HandleUserNickChange(IRCMessage message)
{
    std::string newNick = message.parameters.at(0);
    std::cout << message.prefix.nick << " changed his nick to " << newNick << std::endl;
}

void IRCClient::HandleUserQuit(IRCMessage message)
{
    std::string text = message.parameters.at(0);
    std::cout << message.prefix.nick << " quits (" << text << ")" << std::endl;
}

void IRCClient::HandleChannelNamesList(IRCMessage message)
{
    std::string channel = message.parameters.at(2);
    std::string nicks = message.parameters.at(3);
    std::cout << "People on " << channel << ":" << std::endl << nicks << std::endl;
}

void IRCClient::HandleNicknameInUse(IRCMessage message)
{
    std::cout << message.parameters.at(1) << " " << message.parameters.at(2) << std::endl;
}

void IRCClient::HandleServerMessage(IRCMessage message)
{
    if( message.parameters.empty() )
        return;

    std::vector<std::string>::const_iterator itr = message.parameters.begin();
    ++itr; // skip the first parameter (our nick)
    for (; itr != message.parameters.end(); ++itr)
        std::cout << *itr << " ";
    std::cout << std::endl;
}