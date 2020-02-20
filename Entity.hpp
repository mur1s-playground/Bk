#ifndef ENTITY_HPP
#define ENTITY_HPP

#include "Vector3.hpp"
#include <vector>
#include <string>

enum entity_type {
	ET_PLAYER,
	ET_ITEM,
	ET_STATIC_ASSET
};

struct entity {
	enum entity_type		et;

	char					name[50];
	unsigned int			name_len;

	struct vector3<float>	position;
	float					orientation;

	unsigned int			model_id;
	unsigned int			model_z;
};

using namespace std;

extern vector<struct entity> entities;

void entity_add(string name, enum entity_type et, unsigned int model_id, unsigned int model_z);

#endif