#include "AssetLoader.hpp"

#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

#include "Util.hpp"

#include "stdio.h"

using namespace std;
using namespace cv;

map<string, unsigned int> assets;

void asset_loader_load_all(struct bit_field *asset_store) {
	vector<string> png_files = get_all_files_names_within_folder("./assets/", "png");
	for (int i = 0; i < png_files.size(); i++) {
		Mat image;
		image = imread("./assets/" + png_files[i], -1);
		printf("name %s, channels: %d\n", png_files[i].c_str(), image.channels());
		int size = image.rows * image.cols * image.channels();
		int size_in_bf = (int)ceilf(size / (float) sizeof(unsigned int));
		int position = bit_field_add_bulk(asset_store, (unsigned int *)image.data, size_in_bf, size);
		assets.try_emplace(png_files[i], position + 1);
	}
}