#include "Entity.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

vector<struct entity> entities;

void entity_add(string name, enum entity_type et, unsigned int model_id, unsigned int model_z) {
    struct entity e;
    e.et = et;
    for (int i = 0; i < name.length() && i < 50; i++) {
        e.name[i] = name[i];
        e.name_len = i;
    }
    e.orientation = (float)(rand() % 360);
    e.model_id = model_id;
    e.model_z = model_z;
    entities.push_back(e);
}