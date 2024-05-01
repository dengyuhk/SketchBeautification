import os.path
import torch
from p_searchInte_module.models import parts_search_graphs,im_network

# def combine_graph(_cate,_part_model_dir,_parts,_target_path):
#     _cate_graph = parts_search_graphs(len(_parts))
#
#     for part_id in range(len(_parts)):
#         folder_path=os.path.join(_part_model_dir,_parts[part_id]+'_ae_128')
#         model_dir = os.path.join(folder_path,[file for file in os.listdir(folder_path) if file.endswith('.pth')][0])
#         check_point=torch.load(model_dir)
#         _cate_graph[part_id].load_state_dict(check_point)
#     _graph_path=os.path.join(_target_path,_cate+'_im.pt')
#     torch.save(_cate_graph, _graph_path)


def split_graph(_cate, _graph_path_dir,_parts, _part_model_dir):
    _graph_path=os.path.join(_graph_path_dir, _cate + '_im.pt')
    _cate_graph = torch.load(_graph_path)

    for part_id in range(len(_parts)):

        part_model= _cate_graph[part_id]

        folder_path = os.path.join(_part_model_dir, _parts[part_id] + '_ae_128')
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        part_model_dir = os.path.join(folder_path, _parts[part_id]  +'.pth')


        torch.save(part_model.state_dict(), part_model_dir)


if __name__ == '__main__':
    BASE_DIR='../../'
    cates=['chair','table','airplane','car','guitar','monitor','lampa','vase','mug','lampc']
    cate_parts = {
        'chair': ['chair_arm', 'chair_back', 'chair_leg', 'chair_seat'],
        'table': ['Table_labelA', 'Table_labelB', 'Table_labelC'],
        'airplane': ['airplane_body', 'airplane_wing', 'airplane_tail', 'airplane_engine'],
        'car': ['Car_labelA', 'Car_labelB', 'Car_labelC'],
        'guitar': ['Guitar_labelA', 'Guitar_labelB', 'Guitar_labelC'],
        'monitor': ['Monitor_labelA', 'Monitor_labelB', 'Monitor_labelC'],
        'lampa': ['LampA_labelA', 'LampA_labelB', 'LampA_labelC'],
        'vase': ['unnamed_part_a', 'unnamed_part_b', 'unnamed_part_c', 'unnamed_part_d'],
        'mug': ['Mug_labelA', 'Mug_labelB'],
        'lampc': ['LampC_labelA', 'LampC_labelC', 'LampC_labelD']
    }
    part_model_dir = 'p_searchInte_module/checkpoint'
    graph_model_dir = BASE_DIR+'trained_models'
    for cate in ['chair']:#cates:
        print('processing {}'.format(cate))
        parts=cate_parts[cate]
        split_graph(cate,graph_model_dir,parts,part_model_dir)



