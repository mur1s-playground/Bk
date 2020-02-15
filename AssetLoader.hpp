#ifndef ASSETLOADER_HPP
#define ASSETLOADER_HPP

#include "BitField.hpp"

#include <map>

using namespace std;

extern map<string, unsigned int> assets;

void asset_loader_load_all(struct bit_field* asset_store);

#endif