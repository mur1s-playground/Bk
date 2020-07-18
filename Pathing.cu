#include "Pathing.hpp"

#include "Entity.hpp"
#include "Map.hpp"

#include "cuda_runtime.h"

#ifndef BRUTE_PATHING

unsigned int pathing_add(struct bit_field* bf_rw, struct bit_field* bf_pathing) {
	struct path p;
	p.resolution = { 400, 400 };

	p.from = { 0.0f, 0.0f };
	p.to = { 0.0f, 0.0f };

	p.pathing_x1y1 = { 0, 0 };
	p.pathing_x2y2 = { 399, 399 };

	p.path_calc_stage = -1;

	p.pathing_data = bit_field_add_bulk_zero(bf_pathing, p.resolution[0] * p.resolution[1]) + 1;

	return bit_field_add_bulk(bf_rw, (unsigned int*)&p, ceil(sizeof(struct path) / (float)sizeof(unsigned int)), sizeof(struct path)) + 1;
}

bool pathing_set(struct bit_field* bf_rw, unsigned int pathing_position, struct vector2<float> from, struct vector2<float> to) {
	struct path* p = (struct path*)&bf_rw->data[pathing_position];
	p->from = to;
	p->to = from;
	p->path_calc_stage = 0;

	int min_x = floorf(min(from[0], to[0]));
	int max_x = floorf(max(from[0], to[0]));

	int min_y = floorf(min(from[1], to[1]));
	int max_y = floorf(max(from[1], to[1]));

	if (max_x - min_x > p->resolution[0]-1 || max_y - min_y > p->resolution[1]-1) {
		printf("invalid path\n");
		return false;
	}

	int overhead_x = p->resolution[0] - 1 - (max_x - min_x);
	int overhead_y = p->resolution[1] - 1 - (max_y - min_y);

	p->pathing_x1y1 = { (unsigned int)max((min_x - (overhead_x / 2)), 0), (unsigned int)max((min_y - (overhead_y / 2)), 0) };
	p->pathing_x2y2 = { p->pathing_x1y1[0] + p->resolution[0] - 1, p->pathing_x1y1[1] + p->resolution[1] - 1 };

	//printf("p: x1 %i y1 %i x2 %i y2 %i from %f %f to %f %f\n", p->pathing_x1y1[0], p->pathing_x1y1[1], p->pathing_x2y2[0], p->pathing_x2y2[1], from[0], from[1], to[0], to[1]);

	bit_field_invalidate_bulk(bf_rw, pathing_position, ceil(sizeof(struct path) / (float)sizeof(unsigned int)));
	return true;
}

void pathing_get(struct bit_field* bf_rw, unsigned int pathing_position, struct bit_field* bf_pathing, struct bit_field* bf_map, int path_calc_stage) {
	struct path* p = (struct path*)&bf_rw->data[pathing_position];

	launch_calculate_pathing_kernel(p, bf_rw->device_data[0], pathing_position, bf_pathing->device_data[0], path_calc_stage,
		gm.map_dimensions,
		bf_map->device_data[0], gm.map_pathable_position);
}

vector2<float> pathing_get_next(struct bit_field* bf_rw, unsigned int pathing_position, struct bit_field* bf_pathing, struct vector2<float> position) {
	struct path* p = (struct path*)&bf_rw->data[pathing_position];

	float delta_x = 0.0f;
	float delta_y = 0.0f;

	vector2<int> min_rc = { 0, 0 };
	int min_val = INT_MAX;

	vector2<unsigned int> position_in_path = { (unsigned int)floorf(position[0]) - p->pathing_x1y1[0], (unsigned int)floorf(position[1]) - p->pathing_x1y1[1] };

	int* pathing_data = (int*)&bf_pathing->data[p->pathing_data];	
	for (int r = -1; r <= 1; r++) {
		for (int c = -1; c <= 1; c++) {
			if (r == 0 && c == 0) continue;
			vector2<int> cur_position_in_path = { (int)position_in_path[0] + c, (int)position_in_path[1] + r};
			if (cur_position_in_path[0] >= 0 && cur_position_in_path[0] < p->resolution[0] && cur_position_in_path[1] >= 0 && cur_position_in_path[1] < p->resolution[1]) {
				
				int cur_val = pathing_data[cur_position_in_path[1] * p->resolution[0] + cur_position_in_path[0]];
				if (cur_val < min_val && cur_val > 0) {
					min_val = cur_val;
					min_rc = { c, r };
				}
			}
		}
	}
	return vector2<float>((float)min_rc[0], (float)min_rc[1]);
}

__global__ void calculate_pathing_kernel(
		const unsigned int* device_data_rw, const unsigned int path_position, unsigned int* device_data_pathing, const int path_calc_stage,
		const struct vector2<unsigned int> map_dimensions_center,
		const unsigned int* device_data_map, const unsigned int map_pathables
	) {

		int i = blockIdx.x * blockDim.x + threadIdx.x;

		struct path* p = (struct path*)&device_data_rw[path_position];

		if (i < p->resolution[0] * p->resolution[1]) {
			int p_row = i / p->resolution[0];
			int p_col = i % p->resolution[0];

			int p_game_row = p->pathing_x1y1[1] + p_row;
			int p_game_col = p->pathing_x1y1[0] + p_col;

			int* path_d = (int*)&device_data_pathing[p->pathing_data];

			if (path_calc_stage == 0) {
				p->path_calc_stage = 0;
				unsigned char* frame_pathable = (unsigned char*)&device_data_map[map_pathables];
				path_d[p_row * p->resolution[1] + p_col] = INT_MAX;

				bool set = false;
				
				if (p_game_row >= map_dimensions_center[1] || p_game_col >= map_dimensions_center[0] || frame_pathable[(int)floorf(p_game_row) * map_dimensions_center[0] + (int)floorf(p_game_col)] == 0) {
					path_d[p_row * p->resolution[0] + p_col] = 0;
					set = true;
				} else if ((int)floorf(p->from[0]) == p_game_col && (int)floorf(p->from[1]) == p_game_row) {
					path_d[p_row * p->resolution[0] + p_col] = 1;
					set = true;
				}
				
				int lc = 0;
				while (p->path_calc_stage == 0 && !set && lc < max(p->resolution[0], p->resolution[1])) {
					int candidate_from = INT_MAX;
					int current = path_d[p_row * p->resolution[0] + p_col];
					for (int dr = -1; dr <= 1; dr++) {
						for (int dc = -1; dc <= 1; dc++) {
							if (dr == 0 && dc == 0) continue;
							if (p_game_row + dr >= p->pathing_x1y1[1] && p_game_row + dr <= p->pathing_x2y2[1] && p_game_col + dc >= p->pathing_x1y1[0] && p_game_col + dc <= p->pathing_x2y2[0]) {
								int cur_val = path_d[(p_row + dr) * p->resolution[0] + p_col + dc];
								if (cur_val < INT_MAX && cur_val > 0 && cur_val < candidate_from) {
									candidate_from = cur_val;
								}
							}
						}
					}
					if (candidate_from < INT_MAX && candidate_from < current) {
						path_d[p_row * p->resolution[0] + p_col] = candidate_from+1;
					}
					__syncthreads();
					lc++;
				}
			} else if (path_calc_stage > 0) {
				int lc = 0;
				while (path_calc_stage > 0 && lc < max(p->resolution[0], p->resolution[1])) {
					int candidate_from = INT_MAX;
					int current = path_d[p_row * p->resolution[0] + p_col];
					for (int dr = -1; dr <= 1; dr++) {
						for (int dc = -1; dc <= 1; dc++) {
							if (dr == 0 && dc == 0) continue;
							if (p_game_row + dr >= p->pathing_x1y1[1] && p_game_row + dr <= p->pathing_x2y2[1] && p_game_col + dc >= p->pathing_x1y1[0] && p_game_col + dc <= p->pathing_x2y2[0]) {
								int cur_val = path_d[(p_row + dr) * p->resolution[0] + p_col + dc];
								if (cur_val > 0 && cur_val < candidate_from) {
									candidate_from = cur_val;
								}
							}
						}
					}
					if (candidate_from < INT_MAX && candidate_from < current) {
						path_d[p_row * p->resolution[0] + p_col] = candidate_from + 1;
					}
					__syncthreads();
					lc++;
				}
			}
		}
}

void launch_calculate_pathing_kernel(const struct path *p,
	const unsigned int* device_data_rw, const unsigned int path_position, unsigned int *device_data_pathing, const int path_calc_stage,
	const struct vector2<unsigned int> map_dimensions_center,
	const unsigned int* device_data_map, const unsigned int map_pathables) {

	cudaError_t err = cudaSuccess;

	int threadsPerBlock = 256;
	int blocksPerGrid = (p->resolution[0]*p->resolution[1] + threadsPerBlock - 1) / threadsPerBlock;

	calculate_pathing_kernel << <blocksPerGrid, threadsPerBlock >> > (device_data_rw, path_position, device_data_pathing, path_calc_stage,
		map_dimensions_center, device_data_map, map_pathables);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed in calulate_pathing_kernel (error code %s)\n", cudaGetErrorString(err));
		}

		cudaDeviceSynchronize();
}
#endif

#ifdef PATHING_DEBUG

#ifndef BRUTE_PATHING

__global__ void draw_gpu_pathing_kernel(
	unsigned int* device_data_rw, const unsigned int pathing_position,
	unsigned int* device_data_pathing,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	struct path* p = (struct path*)&device_data_rw[pathing_position];

	if (i < output_width*output_height) {
		int current_x = i % output_width;
		int current_y = i / output_width;

		int current_game_x = (int)floorf(camera_x1 + current_x * camera_z);
		int current_game_y = (int)floorf(camera_y1 + current_y * camera_z);
		
		if (current_game_x >= p->pathing_x1y1[0] && current_game_x < p->pathing_x2y2[0] && current_game_y >= p->pathing_x1y1[1] && current_game_y < p->pathing_x2y2[1]) {
			int* path_data = (int*)&device_data_pathing[p->pathing_data];
			
			int r = current_game_y - (int)p->pathing_x1y1[1];
			int c = current_game_x - (int)p->pathing_x1y1[0];

			unsigned char* output = (unsigned char*)&device_data_output[output_position];
			
			if (path_data[r * p->resolution[0] + c] == 1 || path_data[r * p->resolution[0] + c] == -1) {
				output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 255;
				output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 255;
				output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 255;
			} else if (path_data[r * p->resolution[0] + c] == INT_MAX) {

			} else if (path_data[r * p->resolution[0] + c] < 0) {
				output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 0;
				output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 0;
				output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = (char)(-path_data[r * p->resolution[0] + c] / 3.0f);
			} else if (path_data[r * p->resolution[0] + c] > 0) {
				output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 0;
				output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = (char)(path_data[r * p->resolution[0] + c] / 3.0f);
				output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 0;
			} else if (path_data[r * p->resolution[0] + c] == 0) {
				output[current_y * (output_width * output_channels) + current_x * output_channels + 0] = 255;
				output[current_y * (output_width * output_channels) + current_x * output_channels + 1] = 0;
				output[current_y * (output_width * output_channels) + current_x * output_channels + 2] = 0;
			}
		}
	}
}

void launch_draw_gpu_kernel(
	unsigned int* device_data_rw, const unsigned int pathing_position,
	unsigned int* device_data_pathing,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter) {

	cudaError_t err = cudaSuccess;

		int threadsPerBlock = 256;
		int blocksPerGrid = (output_width*output_height + threadsPerBlock - 1) / threadsPerBlock;

		draw_gpu_pathing_kernel << <blocksPerGrid, threadsPerBlock >> > (device_data_rw, pathing_position, device_data_pathing,
			device_data_output, output_position, output_width, output_height, output_channels,
			camera_x1, camera_y1, camera_z, tick_counter);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed in draw_particles_kernel (error code %s)\n", cudaGetErrorString(err));
		}
	
}
#else

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

#endif