#ifndef UI_HPP
#define UI_HPP

#include "Vector2.hpp"

#include <string>
#include <map>
#include <vector>
#include "BitField.hpp"

using namespace std;

enum on_click_action_type {
	BAT_NONE,
	BAT_UI,
	BAT_GAMESTART,
	BAT_GAMEEND,
	BAT_QUIT,
	BAT_SELECT,
	BAT_GS,
	BAT_MAPEDITOR_SAVEQUIT
};

enum ui_element_type {
	UET_TEXTFIELD,
	UET_BUTTON,
	UET_SCROLLLIST,
	UET_IMAGE,
	UET_CHECKBOX
};

struct ui_element {
	enum ui_element_type			uet;

	string							name;

	struct vector2<unsigned int>	x1y1;
	struct vector2<unsigned int>	x2y2;

	enum on_click_action_type		ocat;
	string							ocap;

	unsigned int					font_size;

	char							value[51];
};

struct ui {
	string						name;
	unsigned int				background_position;

	int							active_element_id;
	int							active_element_param;
	vector<struct ui_element>	ui_elements;
};

extern string                  ui_active;
extern map<string, struct ui>  uis;

extern map<string,unsigned int> ui_elements_position;
extern unsigned int				ui_fonts_position;

void launch_draw_ui_kernel(const unsigned int* bf_assets_data, const unsigned int background_position, const unsigned int font_position, unsigned int* bf_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels, const unsigned int* bf_rw_data, const unsigned int tick_counter);

void ui_init(struct bit_field* bf_assets, struct bit_field *bf_rw);
void ui_set_active(string name);

bool ui_process_click(struct bit_field* bf_rw, unsigned int x, unsigned int y);
bool ui_process_scroll(struct bit_field* bf_rw, unsigned int x, unsigned int y, int z);
void ui_process_keys(struct bit_field* bf_rw, const unsigned int x, const unsigned int y, const unsigned int sdl_keyval_enum);

int ui_value_get_int_from_eid(struct bit_field* bf_rw, string ui_name, int element_idx, int index);
int ui_value_get_int(struct bit_field* bf_rw, string ui_name, string element_name, int index);
void ui_value_as_config_by_eid(struct bit_field* bf_rw, string ui_name, int element_idx, int index, int value);
void ui_value_as_config(struct bit_field* bf_rw, string ui_name, string element_name, int index, int value);

void ui_textfield_set_int(struct bit_field* bf_rw, string ui_name, string ui_element_name, int value);

string ui_textfield_get_value(struct bit_field* bf_rw, string ui_name, string ui_element_name);
void ui_textfield_set_value(struct bit_field* bf_rw, string ui_name, string ui_element_name, const char value[50]);

void ui_save_fields_to_file(struct bit_field* bf_rw, string ui_name, string ui_fields[], unsigned int field_count, string folder, string filename);

#endif