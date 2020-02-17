#include "Storm.hpp"

#include "Main.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <ctime>

#include <math.h>

using namespace std;

unsigned int			storm_phase_current = 0;
unsigned int			storm_phase_time	= 0;

vector<unsigned int>	storm_phase_start_ticks;
vector<unsigned int>	storm_phase_duration_ticks;
vector<float>			storm_phase_mapratio;

struct storm			storm_last;
struct storm			storm_current;
struct storm			storm_to;

__global__ void draw_storm_kernel(unsigned int *device_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels, const unsigned int camera_crop_x1, const unsigned int camera_crop_y1, const float camera_z, const struct storm storm_current, const struct storm storm_to, const unsigned int storm_alpha, const struct vector3<unsigned char> storm_color) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int current_x = (i % width);
		int current_y = (i / width);

		unsigned char* frame = (unsigned char*)&device_output_data[output_position];

		unsigned int storm_alpha = 50;
		if (sqrtf((camera_crop_x1 + current_x*camera_z - storm_current.x) * (camera_crop_x1+ current_x * camera_z - storm_current.x) + (camera_crop_y1 + current_y * camera_z - storm_current.y) * (camera_crop_y1 + current_y * camera_z - storm_current.y)) >= storm_current.radius) {
			frame[current_y * (width * channels) + current_x * channels] = (255 - storm_alpha)/255.0f * frame[current_y * (width * channels) + current_x * channels] + (storm_alpha/255.0f)			* storm_color[0];
			frame[current_y * (width * channels) + current_x * channels + 1] = (255 - storm_alpha) / 255.0f * frame[current_y * (width * channels) + current_x * channels + 1] + (storm_alpha / 255.0f) * storm_color[1];
			frame[current_y * (width * channels) + current_x * channels + 2] = (255 - storm_alpha) / 255.0f * frame[current_y * (width * channels) + current_x * channels + 2] + (storm_alpha / 255.0f) * storm_color[2];
		}
		unsigned int storm_circle_alpha = 150;
		if (sqrtf((camera_crop_x1 + current_x * camera_z - storm_to.x) * (camera_crop_x1 + current_x * camera_z - storm_to.x) + (camera_crop_y1 + current_y * camera_z - storm_to.y) * (camera_crop_y1 + current_y * camera_z - storm_to.y)) >= storm_to.radius-2.0f &&
			sqrtf((camera_crop_x1 + current_x * camera_z - storm_to.x) * (camera_crop_x1 + current_x * camera_z - storm_to.x) + (camera_crop_y1 + current_y * camera_z - storm_to.y) * (camera_crop_y1 + current_y * camera_z - storm_to.y)) <= storm_to.radius+2.0f
			) {
			frame[current_y * (width * channels) + current_x * channels] = (255 - storm_circle_alpha) / 255.0f * frame[current_y * (width * channels) + current_x * channels] + (storm_circle_alpha / 255.0f) * 255;
			frame[current_y * (width * channels) + current_x * channels + 1] = (255 - storm_circle_alpha) / 255.0f * frame[current_y * (width * channels) + current_x * channels + 1] + (storm_circle_alpha / 255.0f) * 255;
			frame[current_y * (width * channels) + current_x * channels + 2] = (255 - storm_circle_alpha) / 255.0f * frame[current_y * (width * channels) + current_x * channels + 2] + (storm_circle_alpha / 255.0f) * 255;
		}
	}
}

void launch_draw_storm_kernel(unsigned int* device_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels, const unsigned int camera_crop_x1, const unsigned int camera_crop_y1, const float camera_z, const struct storm storm_current, const struct storm storm_to, const unsigned int storm_alpha, const struct vector3<unsigned char> storm_color) {
	cudaError_t err = cudaSuccess;

	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

	draw_storm_kernel<<<blocksPerGrid, threadsPerBlock>>> (device_output_data, output_position, width, height, channels, camera_crop_x1, camera_crop_y1, camera_z, storm_current, storm_to, storm_alpha, storm_color);
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed in draw_storm_kernel (error code %s)\n", cudaGetErrorString(err));
	}
}

void storm_init() {
	storm_phase_start_ticks.emplace_back(1800);
	storm_phase_duration_ticks.emplace_back(1800);
	storm_phase_mapratio.emplace_back(0.75f);

	storm_phase_start_ticks.emplace_back(1800);
	storm_phase_duration_ticks.emplace_back(1800);
	storm_phase_mapratio.emplace_back(0.4f);

	storm_phase_start_ticks.emplace_back(1800);
	storm_phase_duration_ticks.emplace_back(1800);
	storm_phase_mapratio.emplace_back(0.2f);

	storm_phase_start_ticks.emplace_back(1800);
	storm_phase_duration_ticks.emplace_back(1800);
	storm_phase_mapratio.emplace_back(0.0f);

	float storm_radius = std::min(map_dimensions[0], map_dimensions[1]) * storm_phase_mapratio[0]/2.0f;
	int storm_center_max_x = (int)floorf(std::max<float>(map_dimensions[0] - storm_radius, 0.0f));
	int storm_center_max_y = (int)floorf(std::max<float>(map_dimensions[1] - storm_radius, 0.0f));
	
	storm_current.x = (unsigned int)map_dimensions[0] / 2.0f;
	storm_current.y = (unsigned int)map_dimensions[1] / 2.0f;
	storm_current.radius = floorf(std::max<float>(map_dimensions[0], map_dimensions[1]) * std::sqrtf(2)/2.0f);
	storm_last.x = storm_current.x;
	storm_last.y = storm_current.y;
	storm_last.radius = storm_current.radius;
	storm_to.x = (unsigned int) storm_radius + (rand() % (int)(storm_center_max_x - storm_radius));
	storm_to.y = (unsigned int) storm_radius + (rand() % (int)(storm_center_max_y - storm_radius));
	storm_to.radius = storm_radius;
}

void storm_next() {
	storm_phase_time++;
	if (storm_phase_time == storm_phase_start_ticks[storm_phase_current] + storm_phase_duration_ticks[storm_phase_current]) {
		if (storm_phase_current + 1 < storm_phase_start_ticks.size()) {
			storm_phase_current++;
			storm_phase_time = 0;
			storm_current = storm_to;
			storm_last = storm_current;

			float storm_radius_new = std::min(map_dimensions[0], map_dimensions[1]) * storm_phase_mapratio[storm_phase_current] / 2.0f;
			float max_dist_from_last_center = storm_last.radius - storm_radius_new;
			float rand_dist = std::rand() / (float)RAND_MAX * max_dist_from_last_center;
			float rand_angle = std::rand() / (float)RAND_MAX * 2 * std::_Pi;

			storm_to.x = (unsigned int)(storm_last.x + rand_dist * std::cosf(rand_angle));
			storm_to.y = (unsigned int)(storm_last.y + rand_dist * std::sinf(rand_angle));
			storm_to.radius = storm_radius_new;
		}
	}
	if (storm_phase_time > storm_phase_start_ticks[storm_phase_current] && storm_phase_time < storm_phase_start_ticks[storm_phase_current] + storm_phase_duration_ticks[storm_phase_current]) {
			int delta_x = (int)(((storm_phase_time - storm_phase_start_ticks[storm_phase_current]) / (float)storm_phase_duration_ticks[storm_phase_current]) * ((int)storm_to.x - (int)storm_last.x));
			int delta_y = (int)(((storm_phase_time - storm_phase_start_ticks[storm_phase_current]) / (float)storm_phase_duration_ticks[storm_phase_current]) * ((int)storm_to.y - (int)storm_last.y));
			storm_current.x = storm_last.x + delta_x;
			storm_current.y = storm_last.y + delta_y;
			storm_current.radius = storm_last.radius + ((storm_phase_time - storm_phase_start_ticks[storm_phase_current]) / (float)storm_phase_duration_ticks[storm_phase_current]) * (storm_to.radius - storm_last.radius);
	}
}