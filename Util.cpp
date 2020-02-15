#include "Util.hpp"

#include <windows.h>

using namespace std;

vector<string> get_all_files_names_within_folder(string folder, string extension) {
    vector<string> names;
    string search_path = folder + "/*." + extension;
    WIN32_FIND_DATA fd;
    HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                names.push_back(fd.cFileName);
            }
        } while (::FindNextFile(hFind, &fd));
        ::FindClose(hFind);
    }
    return names;
}