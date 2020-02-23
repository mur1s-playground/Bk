#include "TwitchIntegration.hpp"

#include <windows.h>
#include <process.h> 
#include <vector>
#include "Util.hpp"
#include "Player.hpp"

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
        return;
    }

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