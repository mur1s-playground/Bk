#include "AssetList.hpp"

#include <sstream>

#include "AssetLoader.hpp"

#include "Playerlist.hpp"

unsigned int assetlist_id_pos		= 0;
unsigned int assetlist_id_size		= 0;

unsigned int assetlist_name_pos		= 0;
unsigned int assetlist_name_size	= 0;

unsigned int assetlist_count		= 0;

void assetlist_init(struct bit_field* bf_rw) {
	int size = 255 * sizeof(struct assetlist_id_element);
	int size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
	assetlist_id_size = size_in_bf;
	assetlist_id_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;
	ui_value_as_config(bf_rw, "mapeditor_menu", "assetlist_id", 0, assetlist_id_pos);
	ui_value_as_config(bf_rw, "mapeditor_menu", "assetlist_id", 1, 0);

	size = 255 * sizeof(struct playerlist_element);
	size_in_bf = (int)ceilf(size / (float)sizeof(unsigned int));
	assetlist_name_size = size_in_bf;
	assetlist_name_pos = bit_field_add_bulk_zero(bf_rw, size_in_bf) + 1;
	ui_value_as_config(bf_rw, "mapeditor_menu", "assetlist_name", 0, assetlist_name_pos);
	ui_value_as_config(bf_rw, "mapeditor_menu", "assetlist_name", 1, 0);
}

void assetlist_add(struct bit_field* bf_rw, unsigned int id, const char name[50]) {
	stringstream ss_ct;
	ss_ct << id;
	struct assetlist_id_element lpe(ss_ct.str().c_str());
	struct assetlist_id_element* lp = (struct assetlist_id_element*) & bf_rw->data[assetlist_id_pos];
	memcpy(&lp[assetlist_count], &lpe, sizeof(struct assetlist_id_element));
	bit_field_invalidate_bulk(bf_rw, assetlist_id_pos, assetlist_id_size);
	ui_value_as_config(bf_rw, "mapeditor_menu", "assetlist_id", 1, assetlist_count + 1);

	struct playerlist_element ple(name);
	struct playerlist_element* pl = (struct playerlist_element*) & bf_rw->data[assetlist_name_pos];
	memcpy(&pl[assetlist_count], &ple, sizeof(struct playerlist_element));
	bit_field_invalidate_bulk(bf_rw, assetlist_name_pos, assetlist_name_size);
	ui_value_as_config(bf_rw, "mapeditor_menu", "assetlist_name", 1, assetlist_count + 1);

	assetlist_count++;
}