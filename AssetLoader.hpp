#ifndef ASSETLOADER_HPP
#define ASSETLOADER_HPP

#include "BitField.hpp"

#include <map>

struct image_dimensions {
	unsigned int width;
	unsigned int height;
};

using namespace std;

extern map<string, unsigned int> assets;
extern map<string, struct image_dimensions> assets_dimensions;

void asset_loader_load_file(struct bit_field* asset_store, string folder, string filename, unsigned int channels);
void asset_loader_load_folder(struct bit_field* asset_store, string folder);
void asset_loader_load_map(struct bit_field* asset_store, string folder, string map_filename_prefix, unsigned int channels);

void asset_loader_unload_file(struct bit_field* asset_store, string asset_path);
void asset_loader_unload_by_containing(struct bit_field* asset_store, string assetpath_contains);

#endif