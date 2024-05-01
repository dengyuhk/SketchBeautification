import numpy as np
from shutil import copyfile
# from modified_HOG import getHOG_1dims,getHOG_1dims_np
# from models import parts_search_graphs
import os
import pickle
import h5py
import json
import torch
import cv2
import copy
from .utils import resample,mask_padding,stroke2img,part_center,resize_vector,patch_pasting,show_sketch
from matplotlib import pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize

##====================================
# Network Setting
##====================================
import os
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


def network_embeding(_cate,_num_parts, _im,_partId,_p_search_model):
    """
     initializing and loading the well-trained models
    """
    ##
    _im=_im.astype(np.float32)
    ## reload the weights of the aen
    # aes_pt_dir = os.path.join(BASE_DIR, 'trained_models', opt.cate+'_im.pt')


    ##processing input image
    batch_ims = torch.from_numpy(_im[np.newaxis,np.newaxis,:,:])
    batch_ims = batch_ims.to(device)
    z_vector, _ = _p_search_model[_partId](batch_ims, None, None, is_training=False)
    latent_e = z_vector.detach().cpu().numpy()

    return latent_e

##COVERT SKETCH
def sk_reformart(_sk,_num_parts,_cate):
    _canvas=np.zeros((256,256), dtype="uint8")
    sk_combine=copy.deepcopy(_sk)
    for partId in range(_num_parts):
        if sk_combine['pFlags'][partId]==True:
            if _cate =='airplane':
                sk_combine['bbx'][1] = sk_combine['bbx'][3]
                # sk_combine['bbx'][3] = sk_combine['bbx']
            px = sk_combine['bbx'][partId]['x']
            py = sk_combine['bbx'][partId]['y']
            pbbx = sk_combine['bbx'][partId]['bbx']

            if len( sk_combine['im'][partId].shape)==3:
                part_im_ =np.array(Image.fromarray(sk_combine['im'][partId]).convert('L'))
                part_im_[part_im_==255]=0
                part_im_[part_im_>0]=1.0
                sk_combine['im'][partId]=part_im_#.astype('float32')
            c_bbx = part_center(sk_combine['im'][partId])
            ## resize the part_image accordingly
            sk_combine['im'][partId]=sk_combine['im'][partId].astype('float32')
            im_content=(mask_padding(0, sk_combine['im'][partId], c_bbx['bbx'] + 4, c_bbx['x'], c_bbx['y']) * 2).astype('float32')

            if c_bbx['bbx']<=pbbx:
                part_data= cv2.resize(im_content, (pbbx, pbbx), interpolation=cv2.INTER_NEAREST)*4
                part_data = skeletonize(np.asarray(part_data) > 0) * 1
            else:
                part_data = cv2.resize(im_content, (pbbx, pbbx), interpolation=cv2.INTER_AREA)*4
                part_data = skeletonize(np.asarray(part_data) > 0) * 1
            temp=np.zeros((256,256), dtype="uint8")
            ## fill into the empty temp with original size s
            temp=patch_pasting(temp,part_data,px,py,pbbx)
            # plt.figure(112)
            # plt.subplot(221)
            # plt.imshow(sk_combine['im'][partId])
            # plt.subplot(222)
            # plt.imshow(im_content)
            # plt.subplot(223)
            # plt.imshow(part_data)
            # plt.subplot(224)
            # plt.imshow(temp)
            # plt.show()

            sk_combine['im'][partId]=temp
            _canvas[temp>0]=1
        else:
            sk_combine['im'][partId] = np.zeros((256,256))
    sk_combine['full_im']=_canvas
    return sk_combine

##DISTANCE COMPUTING
def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx






# def encode_part_dataset(_path):
#     # file_list=os.listdir(_path)
#     for part_name in os.listdir(data_path):
#         if part_name.endswith('_sketch.pt'):
#             print(part_name)
#             part_im_path=os.path.join(data_path,part_name)
#             part_ims= torch.load(part_im_path)['train']['ims'].squeeze()
#             part_im_num=int(len(part_ims)/2)
#
#             feature_list_np = np.zeros((1, 324))  # =np.zeros()
#             for file_id in range(0,part_im_num):
#                 print('processing the [{}/{}]'.format(file_id,part_im_num))
#                 part_im= part_ims[file_id]
#                 feature = getHOG_1dims_np(part_im).squeeze()
#                 feature = feature[np.newaxis, :]
#                 feature_list_np = np.vstack((feature_list_np, feature))
#
#             dict = { 'vecs': feature_list_np}
#             feature_path='data/'+part_name.split('_sketch.pt')[0]+'.pkl'
#             with open(feature_path, 'wb') as fh:
#                 pickle.dump(dict, fh, pickle.HIGHEST_PROTOCOL)
#         else:
#             pass

def read_json_images(_path,_num_parts):
    im_zero= np.zeros((128,128))
    with open(_path) as json_file:
        data = json.load(json_file)
    sk_im = []
    sk_vector = []
    sk_bbxs = []
    sk_pFlags = []
    for pId in range(_num_parts):
        exist_flag = [stroke_e['stroke_class'] for stroke_e in data if stroke_e['stroke_class'] == pId]
        sk_list_arc = []
        if len(exist_flag) == 0:
            pass
        else:
            for stroke_e in data:
                ele_list = []
                if stroke_e['stroke_class'] == pId:
                    for ele in stroke_e['path']:
                        ele_resize = [ele_e * 256 / 512 for ele_e in ele]
                        if ele_resize not in ele_list:  # remove the duplicate elements
                            ele_list.append(ele_resize)
                    p_list_arc = resample(np.array(ele_list), inc=10)
                    sk_list_arc.append(p_list_arc)
                else:
                    pass

        if len(sk_list_arc) != 0:
            img = stroke2img(sk_list_arc, 256)[:, :, 0]
            im = (img != 255)
            _part_data = im * 1
            bbx = part_center(_part_data)
            bias = 0
            _data_temp_e_ = (mask_padding(bias, _part_data, bbx['bbx'] + 4, bbx['x'], bbx['y']) * 2).astype('float32')
            vec_resize = resize_vector(sk_list_arc, [bbx['x'], bbx['y']], bbx['bbx'] + 4, _data_temp_e_,
                                       128)
            img_vector = (stroke2img(vec_resize, 128)[:, :, 0] != 255) * 1
            sk_pFlags.append(True)
            sk_bbxs.append(bbx)
            sk_vector.append(vec_resize)
            sk_im.append(img_vector)
        else:
            sk_pFlags.append(False)
            sk_bbxs.append(None)
            sk_im.append(im_zero)
            sk_vector.append([])
    sk_data = {'im': sk_im, 'vec': sk_vector, 'bbx': sk_bbxs, 'pFlags': sk_pFlags}


    return sk_data


def compute_for_query(_path,cate,_data_path,_parts,_p_search_model):

    cate_folder_path=os.path.join(_path,cate)
    for query_im_name in os.listdir(cate_folder_path):
        if query_im_name.endswith('.png'): continue
        print(query_im_name)
        query_json_path=os.path.join(cate_folder_path,query_im_name)
        sk_data=read_json_images(query_json_path,len(_parts))

        ##READ FEATURE CODES FILE
        file = open(os.path.join('data', cate + '_search_latent_im' + '.pkl'), 'rb')
        codes_cate_file = pickle.load(file)
        file.close()

        if cate=='airplane':
            sk_data['pFlags'][1]=sk_data['pFlags'][3]
            sk_data['pFlags'][3]=False
        for partId in range(len(_parts)):
            print(partId)
            if sk_data['pFlags'][partId]==False:continue

            codes_part=codes_cate_file[partId]
            query_part_im= sk_data['im'][partId]
            # parts_dict=_parts[partId]


            ## pre-compute feature dict and im dict
            feature_codes_part = codes_part['latent']
            imgs_part=codes_part['gt']
            # candidates_ims_path=os.path.join(_data_path,parts_dict+'_sketch.pt')
            # part_ims = torch.load(candidates_ims_path)['train']['ims'].squeeze()
            # part_im_num = int(len(part_ims) / 2)
            # part_ims_=part_ims[0:part_im_num,:]

            query_feature = network_embeding(cate,len(_parts), query_part_im,partId,_p_search_model)
            # query_feature = query_feature[np.newaxis, :]
            dist_mtx = pdist_np(query_feature, feature_codes_part)
            indices = np.argsort(dist_mtx, axis=1)
            top1_id = indices[0][0]
            top_part_im= imgs_part[top1_id]
            sk_data['im'][partId]=top_part_im

            # plt.title(_parts[partId])
            # plt.imshow(top_part_im)
            # plt.show()

        sk_data_reform = sk_reformart(sk_data, len(_parts),cate)
        candidate_im=(1-sk_data_reform['full_im'])
        uint_img = np.array(candidate_im * 255).astype('uint8')
        grayImage = cv2.cvtColor(cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
        top1_target_path=os.path.join(_path,cate,query_im_name.split('.json')[0]+'_candidate.png')

        # plt.subplot(121)
        # plt.imshow(query_part_im)
        # plt.subplot(122)
        # plt.imshow(grayImage)
        # plt.show()
        cv2.imwrite(top1_target_path,grayImage)

        # copyfile(top1_source_path,top1_target_path)






if __name__ == '__main__':
    ##todo: 1. encode all the images to Hog
    BASE_DIR='../../../'
    data_path=BASE_DIR+'data/sketch_out'
    # encode_part_dataset(data_path)
    #
    ##todo: 2. load query and compute the top1
    cates = ['airplane', 'chair', 'table', 'lampa', 'lampc', 'car', 'monitor', 'guitar', 'vase', 'mug']
    cate_parts={'airplane':['airplane_body', 'airplane_wing','airplane_tail','airplane_engine'],
                'chair':['chair_arm', 'chair_back', 'chair_leg', 'chair_seat'],
                'table':['Table_labelA','Table_labelB','Table_labelC'],
                'lampa':['LampA_labelA','LampA_labelB','LampA_labelC'],
                'lampc':['LampC_labelA','LampC_labelC','LampC_labelD'],
                'car':['Car_labelA','Car_labelB','Car_labelC'],
                'monitor':['Monitor_labelA','Monitor_labelB','Monitor_labelC'],
                'guitar':['Guitar_labelA','Guitar_labelB','Guitar_labelC'],
                'vase':['unnamed_part_a','unnamed_part_b','unnamed_part_c','unnamed_part_d'],
                'mug':['Mug_labelA','Mug_labelB'],
                   }
    query_path = 'query'
    for cate in cates:
        print(cate)
        parts=cate_parts[cate]

        aes_pt_dir = os.path.join(BASE_DIR, 'trained_models', cate + '_im.pt')
        p_search_model = torch.load(aes_pt_dir)

        if use_cuda:
            for i in range(len(parts)):
                p_search_model[i] = p_search_model[i].to(device)

        compute_for_query(query_path,cate,data_path,parts,p_search_model)


    # feature=getHOG_1dims('query/airplane40.json_ori.png')
    #
    # print()

    # hog = cv2.HOGDescriptor()
    # im = cv2.imread('query/airplane40.json_ori.png')
    # h = hog.compute(im)
    # print()