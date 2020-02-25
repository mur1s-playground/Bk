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

#include <iostream>
#include <algorithm>
#include "IRCSocket.h"
#include "IRCClient.h"
#include "IRCHandler.h"

std::atomic<bool> player_reg_active = true;
std::string cache_dir;
std::vector<std::string> players;

std::vector<std::string> split(std::string const& text, char sep)
{
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;
    while ((end = text.find(sep, start)) != std::string::npos)
    {
        tokens.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    tokens.push_back(text.substr(start));
    return tokens;
}

bool IRCClient::InitSocket()
{
    return _socket.Init();
}

bool IRCClient::Connect(char* host, int port)
{
    return _socket.Connect(host, port);
}

void IRCClient::Disconnect()
{
    _socket.Disconnect();
}

bool IRCClient::SendIRC(std::string data)
{
    data.append("\n");
    return _socket.SendData(data.c_str());
}

bool IRCClient::Login(std::string nick, std::string user, std::string password)
{
    _nick = nick;
    _user = user;

    if (SendIRC("HELLO"))
    {
        if (!password.empty() && !SendIRC("PASS "+password))
            return false;
        if (SendIRC("NICK " + nick))
            if (SendIRC("USER " + user + " 8 * :Cpp IRC Client"))
                return true;
    }

    return false;
}

void IRCClient::ReceiveData()
{
    std::string buffer = _socket.ReceiveData();

    std::string line;
    std::istringstream iss(buffer);
    while(getline(iss, line))
    {
        if (line.find("\r") != std::string::npos)
            line = line.substr(0, line.size() - 1);
        Parse(line);
    }
}

void IRCClient::Parse(std::string data)
{
    std::string original(data);
    IRCCommandPrefix cmdPrefix;

    // if command has prefix
    if (data.substr(0, 1) == ":")
    {
        cmdPrefix.Parse(data);
        data = data.substr(data.find(" ") + 1);
    }

    std::string command = data.substr(0, data.find(" "));
    std::transform(command.begin(), command.end(), command.begin(), towupper);
    if (data.find(" ") != std::string::npos)
        data = data.substr(data.find(" ") + 1);
    else
        data = "";

    std::vector<std::string> parameters;

    if (data != "")
    {
        if (data.substr(0, 1) == ":")
            parameters.push_back(data.substr(1));
        else
        {
            size_t pos1 = 0, pos2;
            while ((pos2 = data.find(" ", pos1)) != std::string::npos)
            {
                parameters.push_back(data.substr(pos1, pos2 - pos1));
                pos1 = pos2 + 1;
                if (data.substr(pos1, 1) == ":")
                {
                    parameters.push_back(data.substr(pos1 + 1));
                    break;
                }
            }
            if (parameters.empty())
                parameters.push_back(data);
        }
    }

    if (command == "ERROR")
    {
        std::cout << original << std::endl;
        Disconnect();
        return;
    }

    if (command == "PING")
    {
        std::cout << "Ping? Pong!" << std::endl;
        SendIRC("PONG :" + parameters.at(0));
        return;
    }

    IRCMessage ircMessage(command, cmdPrefix, parameters);

    // Default handler
    int commandIndex = GetCommandHandler(command);
    if (commandIndex < NUM_IRC_CMDS)
    {
        IRCCommandHandler& cmdHandler = ircCommandTable[commandIndex];
        (this->*cmdHandler.handler)(ircMessage);
    }
    else if (_debug) {
        //std::cout << "debug msg " << original << std::endl;
        size_t prefix_end_pos = original.find_first_of(' ');
        if (prefix_end_pos != std::string::npos) {
            std::string prefix = original.substr(0, prefix_end_pos);
            size_t nick_end_pos = original.find_first_of(' ', prefix_end_pos + 1);
            if (nick_end_pos != std::string::npos) {
                std::string nick = original.substr(prefix_end_pos + 2, nick_end_pos - prefix_end_pos - 1);
                std::string rest = original.substr(nick_end_pos+1);

                size_t message_start_pos = original.find_first_of(':', nick_end_pos + 1);
                if (message_start_pos != std::string::npos) {
                    //std::cout << "PREFIX:" << prefix << std::endl;

                    size_t nick_end_part_pos = nick.find_first_of('!');

                    if (nick_end_part_pos != std::string::npos) {
                        nick = nick.substr(0, nick_end_part_pos);
                        std::cout << "NICK:" << nick << std::endl;

                        std::string msg = original.substr(message_start_pos + 1);
                        std::cout << "MSG:" << msg << std::endl << std::endl;

                        if (player_reg_active) {
                                if (msg == "!add" || msg == "!play") {
                                    if (std::count(players.begin(), players.end(), nick) == 0) {
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
                                        } else {
                                            dwPos = SetFilePointer(hAppend, 0, NULL, FILE_END);
                                            int nicklen_bytes = (nick.length() + 1);
                                            std::string linetoadd = nick + "\n";
                                            LockFile(hAppend, dwPos, 0, nicklen_bytes, 0);
                                            WriteFile(hAppend, linetoadd.c_str(), nicklen_bytes, &dwBytesWritten, NULL);
                                            UnlockFile(hAppend, dwPos, 0, nicklen_bytes, 0);
                                            CloseHandle(hAppend);
                                            players.push_back(nick);
                                        }
                                    }
                                }
                        }

                        size_t bits_pos = prefix.find("bits=");
                        if (bits_pos != std::string::npos && prefix.length() > bits_pos+4) {
                            std::string bits = prefix.substr(bits_pos + 4);
                            size_t bits_end_pos = bits.find_first_of(';');
                            if (bits_end_pos != std::string::npos) {
                                bits = bits.substr(0, bits_end_pos);
                                std::cout << "BITS:" << bits << std::endl;

                                byte buffer[4096];
                                DWORD dwBytesWritten, dwPos;
                                std::string filepath = cache_dir + "\\bits.txt";
                                HANDLE hAppend = CreateFile(TEXT(filepath.c_str()),
                                    FILE_APPEND_DATA,         // open for writing
                                    FILE_SHARE_READ,          // allow multiple readers
                                    NULL,                     // no security
                                    OPEN_ALWAYS,              // open or create
                                    FILE_ATTRIBUTE_NORMAL,    // normal file
                                    NULL);                    // no attr. template
                                if (hAppend == INVALID_HANDLE_VALUE) {
                                    printf("could not open bits.txt");
                                } else {
                                    dwPos = SetFilePointer(hAppend, 0, NULL, FILE_END);
                                    int nicklen_bytes = (nick.length() + 1 + bits.length() + 2);
                                    std::string linetoadd = nick + ":" + bits + ":";
                                    bool add = false;
                                    if (msg == "!shield") {
                                        linetoadd += "s\n";
                                        add = true;
                                    } else if (msg == "!bandage") {
                                        linetoadd += "b\n";
                                        add = true;
                                    }
                                    if (add) {
                                        LockFile(hAppend, dwPos, 0, nicklen_bytes, 0);
                                        WriteFile(hAppend, linetoadd.c_str(), nicklen_bytes, &dwBytesWritten, NULL);
                                        UnlockFile(hAppend, dwPos, 0, nicklen_bytes, 0);
                                    }
                                    CloseHandle(hAppend);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Try to call hook (if any matches)
    CallHook(command, ircMessage);
}

void IRCClient::HookIRCCommand(std::string command, void (*function)(IRCMessage /*message*/, IRCClient* /*client*/))
{
    IRCCommandHook hook;

    hook.command = command;
    hook.function = function;

    _hooks.push_back(hook);
}

void IRCClient::CallHook(std::string command, IRCMessage message)
{
    if (_hooks.empty())
        return;

    for (std::list<IRCCommandHook>::const_iterator itr = _hooks.begin(); itr != _hooks.end(); ++itr)
    {
        if (itr->command == command)
        {
            (*(itr->function))(message, this);
            break;
        }
    }
}
