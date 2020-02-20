#ifndef ITEM_HPP
#define ITEM_HPP

#include <vector>
#include "BitField.hpp"

struct item {
	unsigned int	item_id;
	int				item_param;
};

using namespace std;

extern unsigned int         item_models_position;
extern vector<struct model> item_models;

void item_models_init(struct bit_field* bf_assets);

#endif