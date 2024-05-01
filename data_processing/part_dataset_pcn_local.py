import os
import numpy as np
import os.path as osp
import torch
import h5py
import cv2
from matplotlib import pyplot as plt
import random
from PIL import Image
from skimage.morphology import  medial_axis, skeletonize
from scipy.ndimage.morphology import distance_transform_edt
import time
from torchvision import transforms
import torchvision.transforms.functional as F
from skimage.morphology import  medial_axis, skeletonize


def neighbours(x, y, image):
    '''Return 8-neighbours of point p1 of picture, in order'''
    i = image
    x1, y1, x_1, y_1 = x + 1, y - 1, x - 1, y + 1
    # print ((x,y))
    return [i[y1][x], i[y1][x1], i[y][x1], i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def zhangSuen(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and  # (Condition 0)
                        P4 * P6 * P8 == 0 and  # Condition 4
                        P2 * P4 * P6 == 0 and  # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):  # Condition 1
                    changing1.append((x, y))
        for x, y in changing1: image[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and  # (Condition 0)
                        P2 * P6 * P8 == 0 and  # Condition 4
                        P2 * P4 * P8 == 0 and  # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):  # Condition 1
                    changing2.append((x, y))
        for x, y in changing2: image[y][x] = 0
        # print changing1
        # print changing2
    return image


def generate_single(data):
    mat = (data > 0).tolist()
    after = zhangSuen(mat)
    after = np.array(after)

    new_data = data * after

    return new_data

pass



class PartDatasetPCN:
    """
    input args:
    output args: num_parts, self_cache[part_label]=[points,normalized_points,existFlag]
    """
    def __init__(self, root, transform_flag=False, class_choice='chair', split='train',):

        self.datapath = root
        self.phase = split
        self.transform_flag=transform_flag
        self.cate = class_choice
        self.part_img_none=np.zeros((256,256))

        datasets = ['PartFPN']  ## we first utilize the global
        cates = {'airplane': {'Huang': 'airplanes', 'psbFPN': 'psbAirplane', 'ShapeFPN': 'shapeAirplane','PartFPN':'Airplane'},
                 'chair': {'psbFPN': 'psbChair', 'ShapeFPN': 'shapeChair', 'Huang': 'chairs',
                           'ShapeFPN_ALL_ROTATE': 'Chair', 'PartFPN': 'Chair', 'PartFPN_Specified_Part': 'Chair', },
                 'vase': {'PartFPN': 'psbVase'},
                 'lampa': {'PartFPN': 'LampA'},
                 'lampc': {'PartFPN': 'LampC'},
                 'mug': {'PartFPN': 'Mug'},
                 'monitor': {'PartFPN': 'Monitor'},
                 'car': {'PartFPN': 'Car'},
                 'table': {'PartFPN': 'Table'},
                 'guitar': {'PartFPN': 'Guitar'},
                 }
        cate_parts = {'chair': 4, 'vase': 4, 'lampa': 3, 'lampc': 3, 'mug': 2, 'monitor': 3, 'airplane': 4, 'car': 3,
                      'table': 3, 'guitar': 3, }
        parts = {'airplane': ['ptA', 'ptB', 'ptC', 'ptD'],
                 'chair': ['chair_arm', 'chair_back', 'chair_leg', 'chair_seat'],
                 'vase': ['ptA', 'ptB', 'ptC', 'ptD'],
                 'lampa': ['ptA', 'ptB', 'ptC'],
                 'lampc': ['ptA', 'ptB', 'ptC'],
                 'mug': ['ptA', 'ptB'],
                 'monitor': ['ptA', 'ptB', 'ptC'],
                 'car': ['ptA', 'ptB', 'ptC'],
                 'table': ['ptA', 'ptB', 'ptC'],
                 'guitar': ['ptA', 'ptB', 'ptC'],
                 }
        # self.pt_dir_all =  os.path.join(self.datapath, '{}_{}_{}.pt'.format(self.cate, 'sketch', 'ae_all'))
        self.pt_dir = os.path.join(self.datapath, '{}_{}_{}_local.pt'.format(self.cate, 'sketch', 'pc'))
        self.num_parts = cate_parts[self.cate]
        if osp.exists(self.pt_dir):
            self._processed_data =torch.load(self.pt_dir)
        else:
            ##todo: all the datsets
            if self.phase == 'train':
                ## for embedding all parts in a single .pt file
                self._processed_data=self._process_datasets_combination(datasets, cates, parts)
            else:
                ## for embedding all parts in a single .pt file
                self._processed_data=self._process_datasets_combination(datasets, cates, parts)

        if self.transform_flag==True:
            self.trans = transforms.Compose([
                transforms.RandomApply([transforms.RandomAffine(degrees=0,translate=(0.02, 0.02), scale=(0.95, 1.05),resample=0)],p=0.99),
                transforms.ToTensor(),
            ])
        else:
            pass




    def __getitem__(self, index):
        if self.phase == 'predict':
            data = self._processed_data['train']['imgs'][index]
            flag = self._processed_data['train']['flags'][index]
        else:
            data = self._processed_data[self.phase]['imgs'][index]
            flag = self._processed_data[self.phase]['flags'][index]

        if self.transform_flag==True:

            # plt.figure(2)
            # ids=0
            trans_data=[]
            gt_data=[]
            for part in data:
                im_data=F.to_pil_image(part*255,mode='RGB')
                # plt.figure(1,figsize=(10,8))
                # plt.subplot(2,2,1)
                # plt.title('original')
                # plt.axis('off')
                # plt.imshow(part[:,:,0])

                tran_data = self.trans(im_data)
                # plt.subplot(2,2,2)
                # plt.title('transformed')
                # plt.axis('off')
                # plt.imshow(tran_data[0])

                tran_data[tran_data >0]=1.0
                # sketch_data = tran_data[0]*1.0


                ## closing opertion doesn't work
                # kernel = np.ones((3, 3), np.uint8)
                # closing = cv2.morphologyEx(sketch_data.numpy(), cv2.MORPH_CLOSE, kernel)  # 闭运算
                # ## registration
                # plt.subplot(2,2,3)
                # plt.title('0/1')
                # plt.axis('off')
                # plt.imshow(tran_data[0])
                # #
                # # # #Thin the sketch
                # sketch_data = skeletonize(tran_data[0].numpy()*1.0)
                # plt.subplot(2,2,4)
                # plt.axis('off')
                # plt.title('skeletonized')
                # plt.imshow(sketch_data)
                # plt.show()
                # tran_data[0] = torch.tensor(sketch_data)




                gt_data.append(torch.FloatTensor(part))
                trans_data.append(torch.transpose(torch.transpose(tran_data,0,2),0,1))

            # print()


        return gt_data,trans_data,flag

    def __len__(self):
        if self.phase == 'predict':
            return len(self._processed_data['train']['imgs'])

        else:
            return len(self._processed_data[self.phase]['imgs'])


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



    def obtain_bbx(self,img):

        ind_mask_e_0 = np.nonzero(img.any(axis=0))[0]
        xmin = ind_mask_e_0[0]
        xmax = ind_mask_e_0[-1]
        ind_mask_e_1 = np.nonzero(img.any(axis=1))[0]
        ymin = ind_mask_e_1[0]
        ymax = ind_mask_e_1[-1]
        # bbx = np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])
        img[ymin:ymax+1, xmin:xmax+1] = 1
        return img  # bbx

    def _process_datasets_combination(self,_datasets, _cates, _parts):
        # raw_data = {}
        imgs=[]
        flags=[]

        _parts_com = _parts[self.cate]
        """#which part"""
        _partData_edge_all = []
        for _dataset in _datasets:
            _cate = _cates[self.cate][_dataset]
            self.hdf5_dir = osp.join(self.datapath, _dataset, '{}_{}_all.hdf5'.format(_cate, self.phase ))

            raw_data_ = h5py.File(self.hdf5_dir, 'r')['ztest']

        for obect_index in range(0,len(raw_data_)):
            parts_imgs = []
            parts_flag=[]
            print('[{}\{}]'.format(obect_index,len(raw_data_)))
            # if obect_index%100==0:
            #     print('processing the {}th object'.format(obect_index))
            object=np.array(raw_data_[str(obect_index)])

            for p_id in range(self.num_parts):
                part_index = np.where(object == p_id+1)
                if len(part_index[0]) > 1:
                    object_=object.copy()
                    object_[np.where(object_ != p_id+1)] = 0
                    object_[np.where(object_ == p_id+1)] = 1
                    part_img=object_

                    threshold = len(np.nonzero(part_img >= 1)[0])
                    if threshold < 30:
                        part_img = self.part_img_none
                        # part_img_norm = self.part_img_none
                        part_img_bbx=self.part_img_none
                        part_cat=np.concatenate((part_img[:,:,np.newaxis],part_img_bbx[:,:,np.newaxis],part_img_bbx[:,:,np.newaxis]),axis=2)
                        # part_img_norm_bbx=self.part_img_none
                        # parts_imgs.append((torch.FloatTensor(part_img),torch.FloatTensor(part_img_bbx), torch.FloatTensor(part_img_norm),torch.FloatTensor(part_img_norm_bbx), is_part_exist))
                        parts_imgs.append(part_cat)
                        parts_flag.append(is_part_exist)
                        # parts_imgs.append((torch.FloatTensor(part_img),  torch.FloatTensor(part_img_norm), is_part_exist))

                        continue
                    temp=part_img.copy()
                    part_img_bbx=self.obtain_bbx(temp)



                    part_cat = np.concatenate((part_img[:, :, np.newaxis], part_img_bbx[:, :, np.newaxis],part_img_bbx[:,:,np.newaxis]), axis=2)
                    is_part_exist=True


                else:
                    part_img = self.part_img_none
                    # part_img_norm=self.part_img_none
                    part_img_bbx = self.part_img_none
                    part_cat = np.concatenate((part_img[:, :, np.newaxis], part_img_bbx[:, :, np.newaxis],part_img_bbx[:,:,np.newaxis]), axis=2)

                    # part_img_norm_bbx = self.part_img_none
                    is_part_exist = False
                # parts_imgs.append((torch.FloatTensor(part_img), torch.FloatTensor(part_img_norm), is_part_exist))
                # parts_imgs.append((torch.FloatTensor(part_img), torch.FloatTensor(part_img_bbx),
                #                    torch.FloatTensor(part_img_norm), torch.FloatTensor(part_img_norm_bbx),
                #                    is_part_exist))
                # plt.subplot(221)
                # plt.imshow(part_img)
                # plt.subplot(222)
                # plt.imshow(part_img_bbx)
                # plt.subplot(223)
                # plt.imshow(part_img_bbx)
                # # plt.subplot(224)
                # # plt.imshow(part_cat[:, :, 0])
                # plt.show()
                # print()
                parts_imgs.append(part_cat)
                parts_flag.append(is_part_exist)
                                   #                    torch.FloatTensor(part_img_norm), torch.FloatTensor(part_img_norm_bbx),
                                   #                    is_part_exist))
            if len(parts_imgs)!=self.num_parts:
                print(obect_index)


            imgs.append(parts_imgs)
            flags.append(parts_flag)
        # processed_data = raw_data#[torch.FloatTensor(_data) for _data in raw_data]

        totalNum = range(len(parts_imgs))
        random.seed(10)
        test_index = random.sample(totalNum, int(1 / 401 * len(parts_imgs)))
        train_index = [el for el in totalNum if el not in test_index]
        ##train val seperate
        ##train
        processed_data_train ={'imgs': imgs,'flags':flags}#processed_data #[processed_data[(index)] for index in train_index]
        ## all data
        imgs_test=[imgs[(index)] for index in test_index]
        flags_test=[flags[(index)] for index in test_index]
        processed_data_test = {'imgs': imgs_test,'flags':flags_test}#[processed_data[(index)] for index in test_index]
        _data_combination = {'train': processed_data_train, 'test': processed_data_test}
        # print('saving to the.pt file')
        current_t=time.time()
        torch.save(_data_combination, self.pt_dir)
        saving_t= time.time()-current_t
        print('saving data takes {} s'.format(saving_t))
        return _data_combination

    def get_number_of_parts(self):
        return self.num_parts


if __name__ == '__main__':
    """[deng]
    Here, we prepare the normlized whole points, and normalized part points,
    with these data, we target at learn a transfromation affine parameters.
    """
    # cate_parts = {'chair', 'airplane','vase', 'lampa', 'lampc', 'mug', 'monitor', 'car',
    #               'table', 'guitar', }
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = PartDatasetPCN(root=os.path.join(BASE_DIR, 'data/sketch_out'), transform_flag=True,
                       class_choice='vase', split='train')

    for i in range(1):
        part_imgs = d[0] #chair:d[6] airplane: d[0]
        # plt.figure(1,figsize=(10,8))


        # m,n=4,4 # part_num, 2 sketches and 2 masks
        # for parts_id in range(m):
        #     plt.subplot(m,n,4*parts_id+1)
        #     plt.axis('off')
        #     im=part_imgs[0][parts_id][:,:,0]
        #     plt.imshow(im)
        #     # plt.show()

        #     plt.subplot(m, n, 4 * parts_id+2)
        #     plt.axis('off')
        #     plt.imshow(part_imgs[1][parts_id][:,:,0])
        #     # plt.show()
        #
        #     plt.subplot(m, n, 4 * parts_id + 3)
        #     plt.axis('off')
        #     plt.imshow(part_imgs[0][parts_id][:, :, 1])
        #
        #     plt.subplot(m, n, 4 * parts_id + 4)
        #     plt.axis('off')
        #     plt.imshow(part_imgs[1][parts_id][:,:,1])
        #     # plt.show()
        # plt.show()
        plt.figure(2)
        m=4
        strokes_ = np.zeros((256, 256), dtype="uint8")
        for parts_id in range(m):
            part = part_imgs[0][parts_id][:,:,0]#gts[parts_id][sample_id][:, :, 0].numpy()
            strokes_[part > 0] = 1
        plt.subplot(121)
        plt.axis('off')
        plt.title('ground_truth')
        plt.tight_layout()
        plt.imshow(strokes_)

        strokes_ = np.zeros((256, 256), dtype="uint8")
        for parts_id in range(m):
            part = part_imgs[1][parts_id][:,:,0]#trans[parts_id][sample_id][:, :, 0].numpy()
            strokes_[part > 0] = 1
        plt.subplot(122)
        plt.axis('off')
        plt.title('random_affine')
        plt.tight_layout()
        plt.imshow(strokes_)
        plt.show()
        print('processing the {}th image'.format(i))
        

        



