#include "TwitchIntegration.hpp"

#include <windows.h>
#include <process.h> 
#include <vector>
#include "Util.hpp"
#include "Player.hpp"
#include <sstream>

string cache_dir = "";

intptr_t irc_process;

void twitch_launch_irc(string cache_dir_, string twitch_name) {
	cache_dir = cache_dir_;
	irc_process = _spawnl(P_NOWAIT, "IRCClient.exe", "IRCClient.exe", "irc.chat.twitch.tv", "6667", "justinfan1337", cache_dir.c_str(), twitch_name.c_str(), NULL);
}

void twitch_terminate_irc() {
    UINT exit_code = 0;
    TerminateProcess((HANDLE)irc_process, exit_code);
}

void twitch_update_players() {
    string file_path = cache_dir + "\\players.txt";
    HANDLE hFile = CreateFile(TEXT(file_path.c_str()), // open One.txt
        GENERIC_READ,             // open for reading
        FILE_SHARE_READ,                        // do not share
        NULL,                     // no security
        OPEN_EXISTING,            // existing file only
        FILE_ATTRIBUTE_NORMAL,    // normal file
        NULL);                    // no attr. template

    if (hFile == INVALID_HANDLE_VALUE) {
        printf("Could not open players.txt.");
    } {

        string file_content = "";

        DWORD  dwBytesRead, dwBytesWritten, dwPos;
        BYTE   buff[4096];
        while (ReadFile(hFile, buff, sizeof(buff), &dwBytesRead, NULL)
            && dwBytesRead > 0) {
            for (int i = 0; i < dwBytesRead; i++) {
                file_content += buff[i];
            }
        }

        CloseHandle(hFile);

        size_t pos = 0;
        size_t fpos = 0;
        while ((fpos = file_content.find('\n', pos)) != string::npos) {
            string name = file_content.substr(pos, fpos - pos);
            name = trim(name);
            player_add(name, PT_HOPE, UINT_MAX);
            pos = fpos + 1;
        }
    }
    /*
    for (int i = 0; i < 100; i++) {
        stringstream ss;
        ss << players.size();
        player_add("mur1_" + ss.str(), PT_HOPE, UINT_MAX);
    }
    */
}

int lines_read = 0;

void twitch_update_bits() {
    string file_path = cache_dir + "\\bits.txt";
    HANDLE hFile = CreateFile(TEXT(file_path.c_str()), // open One.txt
        GENERIC_READ,             // open for reading
        FILE_SHARE_READ,                        // do not share
        NULL,                     // no security
        OPEN_EXISTING,            // existing file only
        FILE_ATTRIBUTE_NORMAL,    // normal file
        NULL);                    // no attr. template

    if (hFile == INVALID_HANDLE_VALUE) {
        printf("Could not open bits.txt.");
    } {

        string file_content = "";

        DWORD  dwBytesRead, dwBytesWritten, dwPos;
        BYTE   buff[4096];
        while (ReadFile(hFile, buff, sizeof(buff), &dwBytesRead, NULL)
            && dwBytesRead > 0) {
            for (int i = 0; i < dwBytesRead; i++) {
                file_content += buff[i];
            }
        }

        CloseHandle(hFile);

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
                    printf("adding bandage money\n");
                    bits_it = bits_bandage.find(name);
                    if (bits_it != bits_bandage.end()) {
                        bits_bandage[name] = bits_it->second + stoi(bits_);
                    } else {
                        bits_bandage.emplace(name, stoi(bits_));
                    }
                } else if (type == "s") {
                    printf("adding shield money\n");
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

    /*
    for (int i = 0; i < 100; i++) {
        stringstream ss;
        ss << players.size();
        player_add("mur1_" + ss.str(), PT_HOPE, UINT_MAX);
    }
    */
}