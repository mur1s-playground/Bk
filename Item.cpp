#include "Item.hpp"

#include "Model.hpp"
#include "Util.hpp"

unsigned int         item_models_position;
vector<struct model> item_models;

void item_models_init(struct bit_field* bf_assets) {
    vector<string> model_cfgs = get_all_files_names_within_folder("./items", "*", "cfg");
    for (int i = 0; i < model_cfgs.size(); i++) {
        struct model m = model_from_cfg(bf_assets, "./items/", model_cfgs[i]);
        item_models.push_back(m);
    }
    vector<struct model> sorted_models;
    int counter = 0;
    struct model empty_model;
    empty_model.id = UINT_MAX;
    while (counter < item_models.size()) {
        for (int i = 0; i < item_models.size(); i++) {
            if (item_models[i].id == counter+50) {
                sorted_models.push_back(item_models[i]);
            }
        }
        if (sorted_models.size() < counter + 1) {
            sorted_models.push_back(empty_model);
        }
        counter++;
    }
    unsigned int size = sorted_models.size() * sizeof(struct model);
    unsigned int size_in_bf = (unsigned int)ceilf(size / (float)sizeof(unsigned int));
    item_models_position = bit_field_add_bulk(bf_assets, (unsigned int*)sorted_models.data(), size, size_in_bf) + 1;
}