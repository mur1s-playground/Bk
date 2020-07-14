#include "Particle.hpp"

#include "Entity.hpp"
#include "Map.hpp"


const int				particles_max		= 1920;
int						particles_max_used	= 0;
struct particle*		particles			= nullptr;

unsigned int				particles_position		= 0;
int							particles_size_in_bf	= 0;

int particle_min_free = 0;

__global__ void draw_particles_kernel(
	const unsigned int* device_data_assets,
	const unsigned int* device_data_rw, const unsigned int particles_position, const unsigned int particles_max_used,
	const struct vector2<unsigned int> map_dimensions_center,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//unsigned int players_count = device_data_players[players_position-1] / (unsigned int)ceilf(sizeof(struct player) / (float)sizeof(unsigned int));
	//struct player* players = (struct player*) &device_data_players[players_position];

	struct particle* particles = (struct particle*)&device_data_rw[particles_position];

	if (i < particles_max_used) {
		if (particles[i].max_lifetime_ticks > 0) {
			int current_channel = i / (output_width * output_height);
			int current_idx = i % (output_width * output_height);
			int current_x = particles[i].position[0];
			int current_y = particles[i].position[1];

			float current_output_x = (current_x - camera_x1) / camera_z;
			float current_output_y = (current_y - camera_y1) / camera_z;

			if (current_output_x >= 0 && current_output_x+2 < output_width && current_output_y >= 0 && current_output_y+2 < output_height) {
				unsigned char* output = (unsigned char*)&device_data_output[output_position];

				for (int dx = 0; dx < 2; dx++) {
					for (int dy = 0; dy < 2; dy++) {
						output[(int)floorf(current_output_y + dy) * (output_width * output_channels) + (int)floorf(current_output_x + dx) * output_channels + 0] = 255;
						output[(int)floorf(current_output_y + dy) * (output_width * output_channels) + (int)floorf(current_output_x + dx) * output_channels + 1] = 255;
						output[(int)floorf(current_output_y + dy) * (output_width * output_channels) + (int)floorf(current_output_x + dx) * output_channels + 2] = 255;
					}
				}
			}
		}
	}
}

void launch_draw_particles_kernel(
	const unsigned int* device_data_assets,
	const unsigned int* device_data_rw, const unsigned int particles_position,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int camera_x1, const unsigned int camera_y1, const float camera_z, const unsigned int tick_counter) {

	cudaError_t err = cudaSuccess;

	if (particles_max_used > 0) {

		int threadsPerBlock = 256;
		int blocksPerGrid = (particles_max_used + threadsPerBlock - 1) / threadsPerBlock;

		draw_particles_kernel << <blocksPerGrid, threadsPerBlock >> > (device_data_assets,
			device_data_rw, particles_position, particles_max_used,
			gm.map_dimensions,
			device_data_output, output_position, output_width, output_height, output_channels,
			camera_x1, camera_y1, camera_z, tick_counter);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed in draw_particles_kernel (error code %s)\n", cudaGetErrorString(err));
		}
	}
}

void particles_init(struct bit_field *bf) {
	unsigned int size = particles_max * sizeof(struct particle);
	particles_size_in_bf = (unsigned int)ceilf(size / (float)sizeof(unsigned int));
	particles_position = bit_field_add_bulk_zero(bf, particles_size_in_bf) + 1;
}

void particles_tick(struct bit_field* bf) {
	particles = (struct particle *) &bf->data[particles_position];
	for (int i = 0; i < particles_max_used; i++) {
		if (particles[i].max_lifetime_ticks > 0) {
			particles[i].max_lifetime_ticks--;

			float dist1 = sqrtf((particles[i].target[0] - particles[i].origin[0]) * (particles[i].target[0] - particles[i].origin[0]) + (particles[i].target[1] - particles[i].origin[1]) * (particles[i].target[1] - particles[i].origin[1])) + 1e-5;

			particles[i].position[0] += particles[i].speed_per_tick * ((particles[i].target[0] - particles[i].origin[0]) / dist1);
			particles[i].position[1] += particles[i].speed_per_tick * ((particles[i].target[1] - particles[i].origin[1]) / dist1);

			float dist2 = sqrtf((particles[i].position[0] - particles[i].origin[0]) * (particles[i].position[0] - particles[i].origin[0]) + (particles[i].position[1] - particles[i].origin[1]) * (particles[i].position[1] - particles[i].origin[1])) + 1e-5;
			if (dist2 >= dist1) {
				particles[i].max_lifetime_ticks = 0;
			}
			if (particles[i].max_lifetime_ticks == 0) {
				if (particle_min_free > i) {
					particle_min_free = i;
				}
				if (i + 1 == particles_max_used) {
					particles_max_used--;
					for (int j = i-1; j >= 0; j--) {
						if (particles[j].max_lifetime_ticks > 0) {
							break;
						}
						particles_max_used--;
					}
				}
			} else {
				if (particles_max_used < i+1) particles_max_used = i+1;
			}
		}
	}
	bit_field_invalidate_bulk(bf, particles_position, particles_size_in_bf);
}

void particles_clear(struct bit_field* bf) {
	particles = (struct particle*)&bf->data[particles_position];
	for (int i = 0; i < particles_max; i++) {
		particles[i].max_lifetime_ticks = 0;
	}
	particle_min_free = 0;
}

void particle_add(struct bit_field *bf, int max_lifetime_ticks, vector2<float> origin, vector2<float> target, float speed_per_tick) {
	particles = (struct particle*)&bf->data[particles_position];
	bool added = false;
	for (int i = particle_min_free; i < particles_max; i++) {
		if (!added) {
			if (particles[i].max_lifetime_ticks <= 0) {
				particles[i].max_lifetime_ticks = max_lifetime_ticks;
				particles[i].origin = origin;
				particles[i].position = origin;
				particles[i].target = target;
				particles[i].speed_per_tick = speed_per_tick;
				if (i+1 > particles_max_used) {
					particles_max_used = i+1;
				}
				added = true;
			}
		} else {
			if (particles[i].max_lifetime_ticks <= 0) {
				particle_min_free = i;
				break;
			}
		}
	}
}