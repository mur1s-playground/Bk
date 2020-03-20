#ifndef ASSETLIST_HPP
#define ASSETLIST_HPP

#include "BitField.hpp"

struct assetlist_id_element {
    assetlist_id_element(const char s[9]) {
        int cur_pos = 0;
        for (int i = 0; i < 9; i++) {
            if (s[i] == '\0') break;
            value[cur_pos] = s[i];
            cur_pos++;
        }
        for (int i = cur_pos; i < 10; i++) {
            value[i] = '\0';
        }
    }
    char value[10];
};

extern unsigned int assetlist_id_pos;

void assetlist_init(struct bit_field* bf_rw);
void assetlist_add(struct bit_field* bf_rw, unsigned int id, const char name[50]);

#endif