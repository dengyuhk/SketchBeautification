##create by deng
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import random
from skimage.morphology import  skeletonize

import cv2



class PartDatasetSearInte(Dataset):

    def __init__(self,_dataPath,_cate,_part_label,_normalized_scale,_phase='train',disorderKey=False,_sk_representation=''):
        'Initialization'
        self.cate = _cate
        self.phase= _phase
        self._normalized_scale=_normalized_scale

        self.sk_representation=_sk_representation
        self.dataPath=_dataPath
        datasets =['PartFPN','PartFPN_Specified_Part']
        cates = {
                 'chair': {'PartFPN': 'Chair', 'PartFPN_Specified_Part': 'Chair'},
                 'table': {'PartFPN': 'Table', 'PartFPN_Specified_Part': 'Table'},
                 'airplane': {'PartFPN': 'Airplane', 'PartFPN_Specified_Part': 'Airplane'},
                 'car':{'PartFPN': 'Car', 'PartFPN_Specified_Part': 'Car'},
                 'guitar':{'PartFPN': 'Guitar', 'PartFPN_Specified_Part': 'Guitar'},
                 'monitor':{'PartFPN': 'Monitor', 'PartFPN_Specified_Part': 'Monitor'},
                 'lampa':{'PartFPN': 'LampA', 'PartFPN_Specified_Part': 'LampA'},
                 'vase': {'PartFPN': 'psbVase', 'PartFPN': 'psbVase'},
                 'mug':   {'PartFPN': 'Mug', 'PartFPN_Specified_Part': 'Mug'},
                 'lampc': {'PartFPN': 'LampC', 'PartFPN_Specified_Part': 'LampC'},
                 }
        parts = {
                 'chair': ['chair_arm', 'chair_back', 'chair_leg', 'chair_seat'],
                 'table': ['Table_labelA', 'Table_labelB', 'Table_labelC'],
                 'airplane': ['airplane_body', 'airplane_wing', 'airplane_tail', 'airplane_engine'],
                 'car':['Car_labelA', 'Car_labelB', 'Car_labelC'],
                 'guitar':['Guitar_labelA','Guitar_labelB','Guitar_labelC'],
                 'monitor': ['Monitor_labelA','Monitor_labelB','Monitor_labelC'],
                 'lampa': ['Lamp_labelA', 'Lamp_labelB', 'Lamp_labelC'],
                 'vase': ['unnamed_part_a', 'unnamed_part_b', 'unnamed_part_c', 'unnamed_part_d'],
                 'mug': ['Mug_labelA', 'Mug_labelB'],
                 'lampc':['Lamp_labelA', 'Lamp_labelC', 'Lamp_labelD'],
                 }
        self.part_name = parts[self.cate][_part_label]
        self.pt_dir = osp.join(self.dataPath, '{}_{}.pt'.format(self.part_name, 'sketch'))

        if self.phase == 'predict':
            self.phase ='train'
        self.hdf5_dir = osp.join(self.dataPath, '{}_{}_{}.hdf5'.format(self.cate, self.phase,'pedges'))
        self.parts_dir=_dataPath
        if osp.exists(self.pt_dir):
            self._processed_data =torch.load(self.pt_dir)
        else:
            ##todo: all the datsets
            if  self.phase=='train':
                self._processed_data =self._process_datasets_parts_normalized(datasets, cates, parts,self._normalized_scale,self.part_name)
            else:
                self._processed_data =torch.load(self.pt_dir)#[self.part_name]
        self.part_scale = self._processed_data['scale']
        self.trans=None


    def __len__(self):
        'Denotes the total number of samples'
        if self.phase == 'predict':
            return len(self._processed_data['train']['ims'])

        else:
            return len(self._processed_data[self.phase]['ims'])

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        print(len(self._processed_data['train']['ims']))
        if self.phase =='predict':
            data_ims = self._processed_data['train']['ims'][index]
            data_points = self._processed_data['train']['points'][index]
            data_values = self._processed_data['train']['values'][index]
            data = {'ims': data_ims, 'points': data_points, 'values': data_values}
            # data = self._processed_data['train'][index]
        else:
            data_ims=self._processed_data[self.phase]['ims'][index]
            data_points = self._processed_data[self.phase]['points'][index]
            data_values = self._processed_data[self.phase]['values'][index]

            data={'ims':data_ims,'points':data_points,'values':data_values}

        return data, index


    def part_normalization(self,_cate,_parts,_part_id,_part_data,_normalized_scale):

        ind_mask_e = np.nonzero(_part_data.any(axis=0))[0]

        centerx_e = int((ind_mask_e[-1] + ind_mask_e[0]) / 2)
        width = ind_mask_e[-1] - ind_mask_e[0]
        ind_mask_e = np.nonzero(_part_data.any(axis=1))[0]
        height = ind_mask_e[-1] - ind_mask_e[0]
        centery_e = int((ind_mask_e[-1] + ind_mask_e[0]) / 2)
        bbx = max(width, height)
        bias = 0

        _data_temp_e = self.mask_padding(bias, _part_data, bbx + 4, centerx_e, centery_e) * 2

        if bbx > _normalized_scale:
            resized = cv2.resize(_data_temp_e, (_normalized_scale, _normalized_scale), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(_data_temp_e, (_normalized_scale, _normalized_scale), interpolation=cv2.INTER_NEAREST)

        _data_temp_e = skeletonize(np.array(resized) > 0) * 1

        return _data_temp_e


    def _process_datasets_parts_normalized(self, _datasets, _cates,_parts,_normalized_scale_,_part_name_specified):
        _data_com = {}
        _data_com_im={}
        _cate = self.cate
        _parts_com = _parts[_cate]
        _normalized_scale = _normalized_scale_


        for _partId in range(len(_parts_com)):
            _data_part = {}
            _part_name = _parts_com[_partId]
            print('processing {}..........'.format(_part_name))
            _partData_edge_all = []
            for _dataset in _datasets:
                _cate = _cates[self.cate][_dataset]
                self.hdf5_dir = osp.join(self.dataPath, _dataset, '{}_{}_{}_{}_no_part_normalization.hdf5'.format(_cate,_part_name, self.phase, 'pedges'))
                raw_data_edge = h5py.File(self.hdf5_dir, 'r')['zedges']  # ['zedges']
                ##todo:=== test the first part
                _partData_edge = raw_data_edge[_part_name]
                _partData_edge_all.append(_partData_edge)

            bbx=_normalized_scale

            bias = 0
            # processed_data = []
            processed_data_im=np.zeros((1,_normalized_scale,_normalized_scale))
            sample_points = np.zeros((1, bbx * bbx, 2), np.uint8)
            sample_values = np.zeros((1, bbx * bbx, 1), np.uint8)
            for partedge_ind in range(len(_partData_edge_all)):

                # _processed_data=[]
                _processed_data_im=np.zeros((1,_normalized_scale,_normalized_scale))
                _sample_points=np.zeros((1,bbx*bbx,2),np.uint8)
                _sample_values=np.zeros((1, bbx * bbx, 1),np.uint8)
                count_=0
                # print('processing {}'.format(partedge_ind))
                for _index in list(_partData_edge_all[partedge_ind].keys()):

                    print('[{}/{}]'.format(count_,len(list(_partData_edge_all[partedge_ind].keys()))))
                    count_= count_ +1
                    img_temp_=np.array(_partData_edge_all[partedge_ind][_index])
                    threshold = len(np.nonzero(img_temp_ >= 1)[0])
                    if threshold < 30: continue
                    part_img_norm = self.part_normalization(_cate, _parts, _partId, img_temp_, bbx)

                    ### sampling points
                    _point_count=0
                    _sample_points_e = np.zeros(( bbx * bbx, 2), np.uint8)
                    _sample_values_e = np.zeros((bbx * bbx, 1), np.uint8)
                    for i in range(0,bbx):
                        for j in range(0, bbx):
                                _sample_points_e[_point_count, 0] = i
                                _sample_points_e[_point_count, 1] = j
                                _sample_values_e[_point_count, 0] = part_img_norm[i, j]
                                _point_count=_point_count+1

                    _processed_data_im = np.vstack((_processed_data_im, part_img_norm[np.newaxis, :, :]))
                    _sample_points=np.vstack((_sample_points,_sample_points_e[np.newaxis,:,:]))
                    _sample_values = np.vstack((_sample_values, _sample_values_e[np.newaxis, :, :]))

                # processed_data = processed_data + _processed_data
                processed_data_im=np.vstack((processed_data_im,_processed_data_im[1:,:,:]))
                sample_points = np.vstack((sample_points, _sample_points[1:, :, :]))
                sample_values = np.vstack((sample_values, _sample_values[1:, :, :]))



            # processed_data = [torch.FloatTensor(_data) for _data in processed_data]
            processed_data_im = processed_data_im[1:,:,:] #for _data_im in processed_data_im]
            processed_data_im=processed_data_im[:,np.newaxis,:,:]
            # processed_data_im = [_data for _data in processed_data_im]
            sample_points = sample_points[1:,:,:]
            sample_values = sample_values[1:, :, :]

            totalNum = range(len(processed_data_im))
            random.seed(10)
            test_index = random.sample(totalNum, int(1 / 401 * len(processed_data_im)))
            # train_index = [el for el in totalNum if el not in test_index]

            ##train val seperate
            # processed_data_im_train = [processed_data_im[(index)] for index in train_index]
            # sample_points_train=[sample_points[(index)] for index in train_index]
            # sample_values_train= [sample_values[(index)] for index in train_index]

            processed_data_im_test = [processed_data_im[(index)] for index in test_index]
            sample_points_test = [sample_points[(index)] for index in test_index]
            sample_values_test = [sample_values[(index)] for index in test_index]

            _data_part = {'train':{'ims': processed_data_im, 'points': sample_points, 'values':sample_values},
                                'test':{'ims': processed_data_im_test, 'points': sample_points_test, 'values':sample_values_test},'scale': bbx}
            _data_com[_part_name] = _data_part

            torch.save(_data_part, os.path.join(self.dataPath,'{}_{}.pt'.format(_part_name,'sketch')))


        return _data_com[_part_name_specified]

    def mask_padding(self,_bias,_data,_bbx,_centerx,_centery):

        _half_patchSize = int(_bbx / 2)

        _width = _data.shape[0]
        _height = _data.shape[1]
        patch = np.zeros((2 * _half_patchSize, 2 * _half_patchSize))

        x_left = _centery - _half_patchSize
        x_right = _centery + _half_patchSize
        y_top = _centerx - _half_patchSize
        y_bottom = _centerx + _half_patchSize

        ## patch is samller than image
        if x_right < _width and x_left >= 0 and y_bottom < _height and y_top >= 0:
            patch = _data[x_left:x_right, y_top:y_bottom]
        elif x_left < 0 and y_top >= 0 and y_bottom < _height:
            patch[(0 - x_left):, :] = _data[0:x_right, y_top:y_bottom]
        elif x_left < 0 and y_top < 0 and y_bottom < _height:
            patch[(0 - x_left):, (0 - y_top):] = _data[0:x_right, 0:y_bottom]
        elif x_left < 0 and y_top >= 0 and y_bottom >= _height:
            patch[(0 - x_left):, 0:(_height - y_top)] = _data[0:x_right, y_top:]
        elif x_right >= _width and y_top >= 0 and y_bottom < _height:
            patch[0:(_width - x_left), :] = _data[x_left:, y_top:y_bottom]
        elif x_right >= _width and y_top < 0 and y_bottom < _height:
            patch[0:(_width - x_left), (0 - y_top):] = _data[x_left:, 0:y_bottom]
        elif x_right >= _width and y_top >= 0 and y_bottom >= _height:
            patch[0:(_width - x_left), 0:(_height - y_top)] = _data[x_left:, y_top:]
        elif x_left >= 0 and x_right < _width and y_top < 0:
            patch[:, (0 - y_top):] = _data[x_left:x_right, 0:y_bottom]
        elif x_left >= 0 and x_right < _width and y_bottom >= _height:
            patch[:, 0:(_height - y_top)] = _data[x_left:(x_right), y_top:]

        ## patch is bigger than image
        elif x_left < 0 and y_top < 0 and y_bottom >= _height and x_right >= _width:
            patch[(0 - x_left):(_width - x_left), (0 - y_top):(_height - y_top)] = _data
        elif y_top < 0 and x_left >= 0 and y_bottom >= _height:
            patch[0:(_width - x_left), (0 - y_top):(_height - y_top)] = _data[x_left:_width, :]
        elif x_left >= 0 and y_top >= 0 and x_right >= _width and y_bottom >= _height:
            patch[0:_width - x_left, 0:_height - y_top] = _data[x_left:_width, y_top:_height]
        elif x_left >= 0 and y_top < 0 and y_bottom < _height and x_right >= _width:
            patch[0:_width - x_left, 0 - y_top:_height - y_top] = _data[x_left:_width, 0:y_bottom]
        elif x_left < 0 and y_top >= 0 and y_bottom >= _height:
            patch[(0 - x_left):_width - x_left, 0:(_height - y_top)] = _data[:, y_top:_height]
        elif x_left < 0 and x_right >= _width and y_top < 0:
            patch[0 - x_left:_width - x_left, 0 - y_top:_height - y_top] = _data[:, 0:y_bottom]
        elif x_left < 0 and y_bottom >= _height and y_top < 0:
            patch[0 - x_left:x_right - x_left, 0 - y_top:_height - y_top] = _data[0:x_right, :]
        elif x_left < 0 and x_right < _width and y_top >= 0 and y_bottom >= _height:
            patch[0 - x_left:x_right - x_left, 0:_height - y_top] = _data[0:x_right, y_top:_height]
        elif x_left < 0 and x_right < _width and y_top < 0 and y_bottom < _height:
            patch[0 - x_left:x_right - x_left, 0 - y_top:y_bottom - y_top] = _data[0: x_right, 0:y_bottom]
        return patch





if __name__ == "__main__":
    # BASE_DIR = '../../'#'
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    normalized_scale_com=128
    dataPath= BASE_DIR+'data/sketch_out'
    d = PartDatasetSearInte(_dataPath=dataPath,_cate='airplane',_normalized_scale=normalized_scale_com,_phase='train',_part_label=0,_sk_representation='')

    for i in range(4):
        test=d[i]
        part_imgs = d[i][0]['ims'].squeeze()
        plt.imshow(part_imgs,'gray')
        plt.show()