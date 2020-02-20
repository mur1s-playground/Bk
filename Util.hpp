#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <string>
#include <map>

using namespace std;

std::string& trim(std::string& str, const std::string& chars = "\t\n\v\f\r ");

vector<string> get_all_files_names_within_folder(string folder, string wildcard, string extension);

map<string, string> get_cfg_key_value_pairs(string folder, string filename);

#endif