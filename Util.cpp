#include "Util.hpp"

#include <windows.h>
#include <fstream>

using namespace std;

std::string& ltrim(std::string& str, const std::string& chars) {
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

std::string& rtrim(std::string& str, const std::string& chars) {
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

std::string& trim(std::string& str, const std::string& chars) {
    return ltrim(rtrim(str, chars), chars);
}

vector<string> get_all_files_names_within_folder(string folder, string wildcard, string extension) {
    vector<string> names;
    string search_path = folder + "/" + wildcard + "." + extension;
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

vector<pair<string, string>> get_cfg_key_value_pairs(string folder, string filename) {
    string filepath = folder + "/" + filename;
    vector<pair<string, string>> result;
        
    ifstream file(filepath);
    string filecontent;
    if (file.is_open()) {
        while (std::getline(file, filecontent)) {
            if (filecontent.size() > 0) {
                size_t last_pos = 0;
                size_t pos = filecontent.find(':');
                if (pos != string::npos) {
                    string name = filecontent.substr(0, pos);
                    name = trim(name);
                    string value = filecontent.substr(pos + 1);
                    value = trim(value);
                    result.push_back(pair<string, string>(name, value));
                    printf("cfg key: %s, value: %s\n", name.c_str(), value.c_str());
                }
            }
        }
    }
    return result;
}

bool dir_exists(const string& dir_in) {
    DWORD ftyp = GetFileAttributesA(dir_in.c_str());
    if (ftyp == INVALID_FILE_ATTRIBUTES)
        return false;

    if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
        return true;

    return false;
}