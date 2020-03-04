#include "AssetLoader.hpp"

#include <vector>
#include <string>

#include "lodepng.h"

#include "Util.hpp"

#include "stdio.h"

using namespace std;

map<string, unsigned int> assets;
map<string, struct image_dimensions> assets_dimensions;

void asset_loader_load_file(struct bit_field* asset_store, string folder, string filename, unsigned int channels) {
	vector<unsigned char> image;
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int size = 0;
	if (channels == 4) {
		lodepng::decode(image, width, height, folder + filename, LCT_RGBA, 8U);
		printf("name %s, width %i, height %i\n", filename.c_str(), width, height);
		size = width * height * channels;
	} else if (channels == 1){
		lodepng::decode(image, width, height, folder + filename, LCT_GREY, 8U);
		printf("name %s, width %i, height %i\n", filename.c_str(), width, height);
		size = width * height * channels;
	}
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
	int position = bit_field_add_bulk(asset_store, (unsigned int*)image.data(), size_in_bf, size);
	assets.try_emplace(folder + filename, position + 1);
	struct image_dimensions img_d;
	img_d.width = width;
	img_d.height = height;
	assets_dimensions.try_emplace(folder + filename, img_d);
}

void asset_loader_load_folder(struct bit_field* asset_store, string folder) {
	vector<string> png_files = get_all_files_names_within_folder(folder, "*", "png");
	for (int i = 0; i < png_files.size(); i++) {
		asset_loader_load_file(asset_store, folder, png_files[i], 4);
	}
}

void asset_loader_load_map(struct bit_field* asset_store, string folder, string map_filename_prefix, unsigned int channels) {
	vector<string> png_files = get_all_files_names_within_folder("./maps/" + folder, map_filename_prefix + "*", "png");
	for (int i = 0; i < png_files.size(); i++) {
		asset_loader_load_file(asset_store, "./maps/" + folder + "/", png_files[i], channels);
	}
}