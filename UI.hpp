#ifndef UI_HPP
#define UI_HPP

#include "Vector2.hpp"

#include <string>
#include <map>
#include <vector>

using namespace std;

enum button_action_type {
	BAT_UI,
	BAT_GAMESTART,
	BAT_QUIT
};

struct button {
	struct vector2<unsigned int>	x1y1;
	struct vector2<unsigned int>	x2y2;
	enum button_action_type			bat;
	string							bap;
};

struct ui {
	string					name;
	unsigned int			background_position;
	vector<struct button>	buttons;
};

extern string                  ui_active;
extern map<string, struct ui>  uis;

void launch_draw_ui_kernel(const unsigned int* bf_assets_data, const unsigned int background_position, unsigned int* bf_output_data, const unsigned int output_position, const unsigned int width, const unsigned int height, const unsigned int channels);

void ui_init(struct bit_field* bf_assets);
void ui_set_active(string name);

void ui_process_click(unsigned int x, unsigned int y);

#endif