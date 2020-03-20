#include "TwitchIntegration.hpp"

#include <windows.h>
#include <process.h> 
#include <vector>
#include "Util.hpp"
#include "Player.hpp"
#include "Playerlist.hpp"
#include <sstream>
#include <time.h>

string cache_dir = "";

HANDLE players_handle;
HANDLE bits_handle;

intptr_t irc_process;

void twitch_launch_irc(string twitch_name) {
    stringstream ss;
    string cache_dir_;
    time_t ltime;
    do {
        ss.clear();
        time(&ltime);
        ss << (long long)ltime;
        cache_dir_ = "cache\\" + ss.str();
    } while (dir_exists(cache_dir_));
    cache_dir = cache_dir_;
    std::string command = "mkdir " + cache_dir;
    system(command.c_str());

    string players_fp = cache_dir + "\\players.txt";
    HANDLE h = CreateFile(TEXT(players_fp.c_str()), GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
    CloseHandle(h);
    string bits_fp = cache_dir + "\\bits.txt";
    h = CreateFile(TEXT(bits_fp.c_str()), GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
    CloseHandle(h);
	
    players_handle = CreateFile(TEXT(players_fp.c_str()), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    bits_handle = CreateFile(TEXT(bits_fp.c_str()), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

	irc_process = _spawnl(P_NOWAIT, "IRCClient.exe", "IRCClient.exe", "irc.chat.twitch.tv", "6667", "justinfan1337", cache_dir.c_str(), twitch_name.c_str(), NULL);
}

void twitch_terminate_irc() {
    UINT exit_code = 0;
    TerminateProcess((HANDLE)irc_process, exit_code);

    CloseHandle(players_handle);
    CloseHandle(bits_handle);
}

void twitch_update_players(struct bit_field* bf_rw) {
    string file_content = "";

    DWORD  dwBytesRead, dwBytesWritten, dwPos;
    BYTE   buff[4096];
    long total_bytes_read = 0;
    while (ReadFile(players_handle, buff, sizeof(buff), &dwBytesRead, NULL)
        && dwBytesRead > 0) {
        for (int i = 0; i < dwBytesRead; i++) {
            file_content += buff[i];
            total_bytes_read++;
        }
    }
    SetFilePointer(players_handle, NULL, NULL, FILE_BEGIN);
    
    size_t pos = 0;
    size_t fpos = 0;
    while ((fpos = file_content.find('\n', pos)) != string::npos) {
        string name = file_content.substr(pos, fpos - pos);
        name = trim(name);
        if (players.size() < players_max) {
            player_add(bf_rw, name, PT_HOPE, UINT_MAX);
        }
        pos = fpos + 1;
    }
    /*
    for (int i = 0; i < 100; i++) {
        stringstream ss;
        ss << players.size();
        string nnnn = "mur1_" + ss.str();
        if (i % 2 == 0) nnnn += "__________________";
        if (players.size() < players_max) {
            player_add(bf_rw, nnnn, PT_HOPE, UINT_MAX);
            //bits_spent.emplace(nnnn, 0);
            //bits_shield.emplace(nnnn, (int)((rand() / (float) RAND_MAX) * 250));
            //bits_bandage.emplace(nnnn, (int)((rand() / (float)RAND_MAX) * 250));
        }
    }
    */
}

int lines_read = 0;

void twitch_update_bits() {
        string file_content = "";

        DWORD  dwBytesRead, dwBytesWritten, dwPos;
        BYTE   buff[4096];
        while (ReadFile(bits_handle, buff, sizeof(buff), &dwBytesRead, NULL)
            && dwBytesRead > 0) {
            for (int i = 0; i < dwBytesRead; i++) {
                file_content += buff[i];
            }
        }
        SetFilePointer(bits_handle, NULL, NULL, FILE_BEGIN);

        size_t pos = 0;
        size_t fpos = 0;
        int cur_lines = 0;
        while ((fpos = file_content.find('\n', pos)) != string::npos) {
            if (cur_lines >= lines_read) {
                string line = file_content.substr(pos, fpos - pos);

                size_t colon_pos = line.find(':');
                string name = trim(line.substr(0, colon_pos));
                
                size_t colon_pos2 = line.find(':', colon_pos + 1);
                string bits_ = trim(line.substr(colon_pos + 1, colon_pos2 - colon_pos - 1));

                string type = trim(line.substr(colon_pos2 + 1));
                
                map<string, int>::iterator bits_it = bits_spent.find(name);

                printf("name   :%s:, bits   :%s:, type   :%s: \n", name.c_str(), bits_.c_str(), type.c_str());

                if (bits_it != bits_spent.end()) {
                } else {
                    bits_spent.emplace(name, 0);
                }
                if (type == "b") {
                    //printf("adding bandage money\n");
                    bits_it = bits_bandage.find(name);
                    if (bits_it != bits_bandage.end()) {
                        bits_bandage[name] = bits_it->second + stoi(bits_);
                    } else {
                        bits_bandage.emplace(name, stoi(bits_));
                    }
                } else if (type == "s") {
                    //printf("adding shield money\n");
                    bits_it = bits_shield.find(name);
                    if (bits_it != bits_shield.end()) {
                        bits_shield[name] = bits_it->second + stoi(bits_);
                    } else {
                        bits_shield.emplace(name, stoi(bits_));
                    }
                }
                lines_read++;
            }
            cur_lines++;
            pos = fpos + 1;
        }
}