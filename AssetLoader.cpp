#include "AssetLoader.hpp"

#include <vector>
#include <string>

#include "lodepng.h"

#include "Util.hpp"

#include "stdio.h"

using namespace std;

map<string, unsigned int> assets;

void asset_loader_load_folder(struct bit_field* asset_store, string folder) {
	vector<string> png_files = get_all_files_names_within_folder(folder, "*", "png");
	for (int i = 0; i < png_files.size(); i++) {
		vector<unsigned char> image;
		unsigned int width = 0;
		unsigned int height = 0;
		lodepng::decode(image, width, height, folder + png_files[i], LCT_RGBA, 8U);
		printf("name %s, width %i, height %i\n", png_files[i].c_str(), width, height);
		int size = width * height * 4;
		int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
		int position = bit_field_add_bulk(asset_store, (unsigned int*)image.data(), size_in_bf, size);
		assets.try_emplace(folder + png_files[i], position + 1);
	}
}

void asset_loader_load_map(struct bit_field* asset_store, string folder, string map_filename_prefix, unsigned int channels) {
	vector<string> png_files = get_all_files_names_within_folder("./maps/" + folder, map_filename_prefix + "*", "png");
	for (int i = 0; i < png_files.size(); i++) {
		vector<unsigned char> image;
		unsigned int width = 0;
		unsigned int height = 0;
		unsigned int size = 0;
		if (channels == 4) {
			lodepng::decode(image, width, height, "./maps/" + folder + "/" + png_files[i], LCT_RGBA, 8U);
			size = width * height * 4;
		} else if (channels == 1){
			lodepng::decode(image, width, height, "./maps/" + folder + "/" + png_files[i], LCT_GREY, 8U);
			size = width * height;
		}
		printf("name %s, width %i, height %i, channels: %i\n", png_files[i].c_str(), width, height, channels);
		int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
		int position = bit_field_add_bulk(asset_store, (unsigned int*)image.data(), size_in_bf, size);
		assets.try_emplace("./maps/" + folder + "/" + png_files[i], position + 1);
	}
}