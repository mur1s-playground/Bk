#include "Pathing.hpp"

#ifdef PATHING_DEBUG

#include "Entity.hpp"
#include "Map.hpp"

#include "cuda_runtime.h"

__global__ void draw_pathing_kernel(
	const unsigned int* device_data_assets,
	const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int entities_count,
	const struct vector2<unsigned int> map_dimensions_center,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	struct entity* entities = (struct entity*)&device_data_rw[entities_position];

	if (i < entities_count) {
		if (entities[i].et == ET_PLAYER) {
			int* params = (int*)&entities[i].params;
			int params_pos = 1;
			for (int ip = 0; ip < 6; ip++) {
				params_pos++;
				params_pos++;
			}
			int path_count = params[params_pos++];
			for (int mp = 0; mp < path_count-1; mp++) {
				int current_x = params[params_pos];
				int current_y = params[params_pos + 1];

				float current_output_x = (current_x - camera_x1) / camera_z;
				float current_output_y = (current_y - camera_y1) / camera_z;

				int current_x2 = params[params_pos + 2];
				int current_y2 = params[params_pos + 3];

				float current_output_x2 = (current_x2 - camera_x1) / camera_z;
				float current_output_y2 = (current_y2 - camera_y1) / camera_z;

				float dir_x = (current_output_x2 - current_output_x);
				float dir_y = (current_output_y2 - current_output_y);

				params_pos++;
				params_pos++;

				for (int p = 0; p < 32; p++) {
					int cur_val_x = (int)floorf(current_output_x + (p / 32.0f) * dir_x);
					int cur_val_y = (int)floorf(current_output_y + (p / 32.0f) * dir_y);
					if (cur_val_x >= 0 && cur_val_x < output_width && cur_val_y >= 0 && cur_val_y < output_height) {
						unsigned char* output = (unsigned char*)&device_data_output[output_position];
						output[cur_val_y * (output_width * output_channels) + cur_val_x * output_channels + 0] = 0;
						output[cur_val_y * (output_width * output_channels) + cur_val_x * output_channels + 1] = 0;
						output[cur_val_y * (output_width * output_channels) + cur_val_x * output_channels + 2] = 0;
					}
				}
			}
		}
	}
}

void launch_draw_pathing_kernel(
	const unsigned int* device_data_assets,
	const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int entities_count,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter) {

	cudaError_t err = cudaSuccess;

	if (entities_count > 0) {

		int threadsPerBlock = 256;
		int blocksPerGrid = (entities_count + threadsPerBlock - 1) / threadsPerBlock;

		draw_pathing_kernel << <blocksPerGrid, threadsPerBlock >> > (device_data_assets,
			device_data_rw, entities_position, entities_count,
			gm.map_dimensions,
			device_data_output, output_position, output_width, output_height, output_channels,
			camera_x1, camera_y1, camera_z, tick_counter);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed in draw_particles_kernel (error code %s)\n", cudaGetErrorString(err));
		}
	}
}

#endif