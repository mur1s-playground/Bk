#ifndef ASSETLOADER_HPP
#define ASSETLOADER_HPP

#include "BitField.hpp"

#include <map>

using namespace std;

extern map<string, unsigned int> assets;

void asset_loader_load_folder(struct bit_field* asset_store, string folder);
void asset_loader_load_map(struct bit_field* asset_store, string folder, string map_filename_prefix, unsigned int channels);

#endif