import os
import json
import time
import numpy as np
import torch
from p_searchInte_module.models import parts_search_graphs
import pickle

import argparse

parser = argparse.ArgumentParser(description='code_generation')
parser.add_argument('--cate', type=str, default='chair')
parser.add_argument('--normalized_scale', type=int, default=128)
parser.add_argument('--save_folder', type=str, default='code_books')
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--root', type=str, default='')
parser.add_argument('--num_parts', type=int, default=4)
parser.add_argument('--.sk_representation', type=str, default='')
opt = parser.parse_args()


use_cuda = torch.cuda.is_available()
device = torch.device('cuda')

def embeding(opt):
    if opt.action == 'generating':
        phase='train'
        print('generating code and image for {}........................'.format(opt.cate))
        folder = opt.save_folder  # '512_d_l2_distance'#'512_d_l2_axis_0.1' # 512_d_l2_axis_0.1

    ##todo: initializing and loading the well-trained models
    ## reload the weights of the aen
    aes_pt_dir = os.path.join(BASE_DIR, 'trained_models', opt.cate+'_im.pt')
    p_search_model = torch.load(aes_pt_dir)

    if use_cuda:
        for i in range(opt.num_parts):
            p_search_model[i] = p_search_model[i].to(device)



    ## for search code book
    code_book = {}
    with torch.no_grad():
        for partId in range(len(opt.part_names)):
            ##load input sketches
            part_name=opt.part_names[partId]
            data_pt_name = opt.root +'/'+ part_name + '_sketch.pt'
            if os.path.exists(data_pt_name):
                data_dict = torch.load(data_pt_name)
                data_dict = data_dict['train']
                data_ims = data_dict['ims'][:]  # NXY
                data_ims = data_ims[:]
                shape_num = len(data_ims)

            ## initlize the neteork
            p_search_model[partId].eval()

            ## feed data
            latent = np.zeros((1, opt.latent_dim))
            chamfer_points=[]
            gt = np.zeros((1, opt.normalized_scale, opt.normalized_scale))
            t0 = 0
            for t in range(20, shape_num + 20, 20):
                if t0 + 20 > shape_num:
                    t = shape_num
                else:
                    pass

                print('{}: processing [{}/{}]'.format(part_name,t0,shape_num))
                batch_ims = data_ims[t0:t].astype(np.float32)
                point_e=[np.asarray(np.nonzero(im.squeeze())).T for im in batch_ims]
                chamfer_points=chamfer_points+point_e
                batch_ims = torch.from_numpy(batch_ims)
                batch_ims = batch_ims.to(device)
                z_vector, _ =p_search_model[partId](batch_ims, None, None, is_training=False)
                latent_e = z_vector.detach().cpu().squeeze()
                latent = np.vstack((latent, latent_e))
                # gt_e = batch_ims.detach().cpu().squeeze()
                # if len(gt_e.shape) == 2:
                #     gt_e = gt_e.unsqueeze(0)
                # gt = np.vstack((gt, gt_e))
                t0=t
            latent = latent[1:, :]
            # gt = gt[1:, :, :]

            data_save = {'latent': latent, 'cham_pt': chamfer_points}
            code_book[partId] = data_save

        if not os.path.exists(os.path.join('../../', 'part_retrieval')): os.makedirs(os.path.join('../../', 'part_retrieval'))
        if not os.path.exists(os.path.join('../../', 'part_retrieval', folder)): os.makedirs( os.path.join('../../', 'part_retrieval', folder))
        predict_path = os.path.join('../../','part_retrieval', folder, '{}_{}_{}.pkl'.format(opt.cate, 'search', 'latent_cham_pt'))
        with open(predict_path, 'wb') as f:
            pickle.dump(code_book, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cates = ['chair', 'table', 'airplane', 'car', 'guitar', 'monitor', 'lampa', 'vase', 'mug', 'lampc']
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

    BASE_DIR = '../../../'
    for cate in cates:
        opt.cate=cate
        # opt.cate='vase'

        root = os.path.join(BASE_DIR, 'data/sketch_out')
        opt.root = root

        opt.num_parts = len(cate_parts[opt.cate])
        opt.part_names=cate_parts[opt.cate]
        ## todo: =========================
        opt.action= 'generating'##'predicting' # 'predicting'
        embeding(opt)