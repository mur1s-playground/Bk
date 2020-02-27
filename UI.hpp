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
	BAT_QUIT,
	BAT_SELECT,
};

enum ui_element_type {
	UET_TEXTFIELD,
	UET_BUTTON,
	UET_SCROLLLIST,
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
	vector<struct ui_element>	ui_elements;
};

extern string                  ui_active;
extern map<string, struct ui>  uis;

extern map<string,unsigned int> ui_elements_position;
extern unsigned int				ui_fonts_position;

void launch_draw_ui_kernel(const unsigned int* bf_assets_data, const unsigned int background_position, const unsigned int font_position, unsigned int* bf_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels, const unsigned int* bf_rw_data, const unsigned int tick_counter);

void ui_init(struct bit_field* bf_assets, struct bit_field *bf_rw);
void ui_set_active(string name);

void ui_process_click(unsigned int x, unsigned int y);
bool ui_process_scroll(struct bit_field* bf_rw, unsigned int x, unsigned int y, int z);
void ui_process_keys(struct bit_field* bf_rw, const unsigned int x, const unsigned int y, const unsigned int sdl_keyval_enum);

void ui_value_as_config(struct bit_field* bf_rw, string ui_name, string element_name, int index, int value);

void ui_textfield_set_int(struct bit_field* bf_rw, string ui_name, string ui_element_name, int value);
void ui_textfield_set_value(struct bit_field* bf_rw, string ui_name, string ui_element_name, char value[50]);
#endif