import os
import time
import math
import random
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from models import im_network

def update_learning_rate(optimizer,scheduler):
    """
    update learning rate (called once every epoch)
    """
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.8)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class IM_AE(object):
    def __init__(self, config):
        # progressive training
        # 1-- (16, 16*16*16)
        # 2-- (32, 16*16*16)
        # 3-- (64, 16*16*16*4)
        self.sample_im_size = config.sample_im_size #sample_vox_size
        self.part_name=config.part_name

        self.input_size = 128#64  # input voxel grid size
        self.load_point_batch_size = 128*128 #16 * 16 * 16 * 4  ### why is different? maybe input data size is the same size
        self.point_batch_size = 128*128#16 * 16 * 16  # train how many points one time
        self.shape_batch_size = 1#32  # train how many shapes one time

        self.ef_dim = 32  # encoder channel 32?
        self.gf_dim = 128  # decoder channel 128?
        self.z_dim = 512#256  # latent code dimension
        self.point_dim = 2#3  # point dimension

        ##setting checkpoint
        self.checkpoint_dir = config.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)

        ## loading traning data
        BASE_DIR ='../../'# '/home/yudeng/data/projects/'
        self.data_dir = BASE_DIR+config.data_dir

        data_pt_name = self.data_dir + config.part_name + '_sketch.pt'
        if os.path.exists(data_pt_name):
            data_dict = torch.load(data_pt_name)#h5py.File(data_pt_name, 'r')
            if (config.train or config.getz):

                data_dict=data_dict['train']
                self.data_points = (data_dict['points'][:].astype( np.float32) )   # spatial normalize to [-0.5, 0.5]
                self.data_values = data_dict['values'][:].astype(np.float32)  # the occupied is 1.
                self.data_ims = data_dict['ims'][:]  # NXY
                self.data_ims=self.data_ims[:]#,np.newaxis,:,:]# NCXY
                # reshape to NCXYZ
                # self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.input_size, self.input_size, self.input_size])

            else:
                data_dict=data_dict['train']
                self.data_points = (data_dict['points'][:].astype(np.float32) )  # spatial normalize to [-0.5, 0.5]
                self.data_values = data_dict['values'][:].astype(np.float32)  # the occupied is 1.
                self.data_ims = data_dict['ims'][:]  # NCXY
               # NCXY

        else:
            print("error: cannot load " + data_pt_name)
            exit(0)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # build model
        self.im_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)  # 32 128 256
        self.im_network.to(self.device)

        # print params
        # for param_tensor in self.im_network.state_dict():
        #     print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
        self.optimizer = torch.optim.Adam(self.im_network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
        self.scheduler = get_scheduler( self.optimizer, config)

        ##pytorch does not have a checkpoint manager, have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)
        self.checkpoint_name = 'IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

        # loss
        def network_loss(G, point_value):
            return torch.mean((G - point_value) ** 2)

        self.loss = network_loss

        # keep everything a power of 2                          ### for what?
        self.cell_grid_size = 1  ### why 4? 256 = 64* 4
        self.frame_grid_size = 128  ### resolution?
        self.real_size = self.cell_grid_size * self.frame_grid_size  # 256, output point-value voxel grid size in testing          ### for what?
        self.test_size = 128  # related to testing batch_size, adjust according to gpu memory size                                  ### batch size related?
        self.test_point_batch_size = self.test_size * self.test_size  # do not change                                 ### this 32*32*32 half volume of a voxel?

        #get coords for training and testing
        self.coords = np.zeros((self.input_size * self.input_size, 2), np.uint8)
        _point_count = 0
        for i in range(0, self.input_size):
            for j in range(0, self.input_size):
                self.coords[_point_count, 0] = i
                self.coords[_point_count, 1] = j
                _point_count = _point_count + 1

        self.sampling_threshold = 0.5  # final marching cubes threshold

    @property
    def model_dir(self):
        return "{}_ae_{}".format(self.part_name, self.input_size)

    def train(self, config):
        # load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        shape_num = len(self.data_ims)
        batch_index_list = np.arange(shape_num)

        print("\n\n----------net summary----------")
        print("training samples   ", shape_num)
        print("-------------------------------\n\n")

        start_time = time.time()
        assert config.epoch == 0 or config.iteration == 0
        training_epoch = config.epoch + int(config.iteration / shape_num)
        batch_num = int(shape_num / self.shape_batch_size)
        point_batch_num = int(self.load_point_batch_size / self.point_batch_size)  # 4096 one time


        for epoch in range(0, training_epoch):
            self.im_network.train()

            np.random.shuffle(batch_index_list)
            avg_loss_sp = 0
            avg_num = 0
            for idx in range(batch_num):

                dxb = batch_index_list[idx * self.shape_batch_size:(idx + 1) * self.shape_batch_size]
                batch_ims = self.data_ims[dxb].astype(np.float32)
                if point_batch_num == 1:
                    point_coord = self.data_points[dxb]
                    point_value = self.data_values[dxb]
                else:
                    which_batch = np.random.randint(point_batch_num)
                    point_coord = self.data_points[dxb,
                                  which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size]
                    point_value = self.data_values[dxb,
                                  which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size]

                batch_ims = torch.from_numpy(batch_ims)
                point_coord = torch.from_numpy(point_coord)
                point_value = torch.from_numpy(point_value)

                batch_ims = batch_ims.to(self.device)
                point_coord = point_coord.to(self.device)
                point_value = point_value.to(self.device)

                self.im_network.zero_grad()
                _, net_out = self.im_network(batch_ims, None, point_coord, is_training=True)
                errSP = self.loss(net_out, point_value)

                errSP.backward()
                self.optimizer.step()

                avg_loss_sp += errSP.item()
                avg_num += 1

                print(" Batch: [%2d/%2d] time: %4.4f, loss_sp: %.6f  lr=  %.9f" % (idx, batch_num, time.time() - start_time, avg_loss_sp / avg_num, self.optimizer.param_groups[0]['lr']))

            update_learning_rate(self.optimizer, self.scheduler)
            print(" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f  lr=  %.9f" % ( epoch, training_epoch, time.time() - start_time, avg_loss_sp / avg_num,self.optimizer.param_groups[0]['lr']))
            print('\n')
            if epoch % 10 == 9:
                self.test_1(config, "train_" + str(self.sample_im_size) + "_" + str(epoch))
            if epoch % 20 == 19:
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                save_dir = os.path.join(self.checkpoint_path,
                                        self.checkpoint_name + str(self.sample_im_size) + "-" + str(epoch) + ".pth")
                self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.max_to_keep
                # delete checkpoint
                if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
                    if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                        os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
                # save checkpoint
                torch.save(self.im_network.state_dict(), save_dir)
                # update checkpoint manager
                self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
                # write file
                checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
                fout = open(checkpoint_txt, 'w')
                for i in range(self.max_to_keep):
                    pointer = (self.checkpoint_manager_pointer + self.max_to_keep - i) % self.max_to_keep
                    if self.checkpoint_manager_list[pointer] is not None:
                        fout.write(self.checkpoint_manager_list[pointer] + "\n")
                fout.close()

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,
                                self.checkpoint_name + str(self.sample_im_size) + "-" + str(epoch) + ".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.max_to_keep
        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        # save checkpoint
        torch.save(self.im_network.state_dict(), save_dir)
        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        # write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer + self.max_to_keep - i) % self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer] + "\n")
        fout.close()


    def test_reconstruction(self,config):
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Network Load SUCCESS")
        else:
            print(" [!] Network Load failed...")
            return

        self.im_network.eval()
        shape_num = 10#len(self.data_ims)
        with torch.no_grad():
            while True:
                plt.figure(1,figsize=(8,6))
                for t in range(shape_num):
                    randNum=np.random.randint(len(self.data_ims))
                    if randNum +shape_num > len(self.data_ims):continue
                    plt.subplot(2, shape_num, t + 1)
                    batch_ims = self.data_ims[t+randNum:t+randNum + 1].astype(np.float32)
                    plt.axis('off')
                    plt.imshow(batch_ims.squeeze())

                    ## reconstruction
                    frame_flag = np.zeros([self.input_size, self.input_size], np.uint8)
                    batch_ims = torch.from_numpy(batch_ims)
                    batch_ims = batch_ims.to(self.device)
                    z_vector, _ = self.im_network(batch_ims, None, None, is_training=False)

                    # get frame grid values
                    point_coord = self.coords[np.newaxis, :, :].astype(np.float32)
                    point_coord = torch.from_numpy(point_coord).to(self.device)

                    _, model_out_ = self.im_network(None, z_vector, point_coord, is_training=False)
                    model_out = model_out_.detach().cpu().numpy()  # [0]
                    x_coords = self.coords[0:self.test_point_batch_size][:, 0]
                    y_coords = self.coords[0:self.test_point_batch_size][:, 1]
                    frame_flag[x_coords, y_coords] = np.reshape((model_out > self.sampling_threshold).astype(np.uint8),
                                                                [self.test_point_batch_size])  # self.sampling_threshold
                    plt.subplot(2, shape_num, shape_num + t + 1)
                    plt.axis('off')
                    plt.imshow(frame_flag * 255)
                plt.tight_layout()
                plt.show()
                print()


    def z2im(self, z,config, name='code_generated'):
        model_float = np.zeros([self.real_size, self.real_size], np.float32)
        self.im_network.eval()
        # frame_flag = np.zeros([self.input_size, self.input_size], np.uint8)
        # t = np.random.randint(len(self.data_ims))
        # batch_ims = self.data_ims[t:t + 1].astype(np.float32)
        # batch_ims = torch.from_numpy(batch_ims)
        # batch_ims = batch_ims.to(self.device)
        # z_vector, _ = self.im_network(batch_ims, None, None, is_training=False)
        z_vector=torch.from_numpy(z).to(self.device)
        # get frame grid values
        point_coord = self.coords[np.newaxis, :, :].astype(np.float32)
        point_coord = torch.from_numpy(point_coord).to(self.device)

        _, model_out_ = self.im_network(None, z_vector, point_coord, is_training=False)
        model_out = model_out_.detach().cpu().numpy()  # [0]
        x_coords = self.coords[0:self.test_point_batch_size][:, 0]
        y_coords = self.coords[0:self.test_point_batch_size][:, 1]
        model_float[x_coords, y_coords] = np.reshape((model_out > self.sampling_threshold).astype(np.uint8),
                                                    [self.test_point_batch_size])  # self.sampling_threshold

        cv2.imwrite(config.sample_dir + "/" + name + ".png", model_float * 255)
        print("[sample]")

    def interpolation(self,config):
        ## load the network weights
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Network Load SUCCESS")
        else:
            print(" [!] Network Load failed...")
            return

        self.im_network.eval()
        import matplotlib.pyplot as plt
        plt.figure(1)
        shape_num = 16#len(self.data_ims) # [t1,t2,shape_num,interval]
        ##aiplane_body [4，5, 0,8] [0,6,10,8] [3,6,10,8],[2,3,182,8], [0,6,388,8],
        # airplane_wing [0,5,7,8],[14,15,0,16] [14,22,0,32]
        # airplane tail [2,11,16,16], [7,11,36,20]
        # mug_a  [0,15,16,16]
        # mug_b  [1,8,0,16]
        # chair_leg [2,3,16,16] ,[24,27,64,32] ,[4,3,16,16],[0,21,16,32] [21, 39, 16,48]
        #chair_back [3,4,20,36] [1,13,20, 36] [13,16,20,0]
        # chair_arm [0,1,36,20]
        #chair_seat [5,6,36,20]
        #car_a [0,1,36,20] [0,11,36,20] [0,4,36,20]
        #car_b [6,8,36,20]
        # guitar_labelC[15,20,36,20]
        t1 = 4
        t2 = 3
        interval = 16
        for t in range(shape_num):
            plt.subplot(4,shape_num/4,t+1)
            batch_ims = self.data_ims[t+interval:t+interval + 1].astype(np.float32)
            plt.axis('off')
            plt.imshow(batch_ims.squeeze())

        im1 = self.data_ims[t1+interval:t1+interval + 1].astype(np.float32)
        im2 = self.data_ims[t2+interval:t2+interval + 1].astype(np.float32)


        im1_np =im1.astype(np.float32)
        im1_gpu= torch.from_numpy(im1_np).to(self.device)
        z_1, _ = self.im_network(im1_gpu, None, None, is_training=False)

        im2_np = im2.astype(np.float32)
        im2_gpu = torch.from_numpy(im2_np).to(self.device)
        z_2, _ = self.im_network(im2_gpu, None, None, is_training=False)

        plt.figure(2,figsize=(8,6))

        inter_num=10
        for i in range(inter_num+1):
            model_float=np.zeros([self.real_size, self.real_size], np.float32)
            weight=i*0.1
            z_interp=weight*z_2+(1-weight)*z_1

            # get frame grid values
            point_coord = self.coords[np.newaxis, :, :].astype(np.float32)
            point_coord = torch.from_numpy(point_coord).to(self.device)

            _, model_out_ = self.im_network(None, z_interp, point_coord, is_training=False)
            model_out = model_out_.detach().cpu().numpy()  # [0]
            x_coords = self.coords[0:self.test_point_batch_size][:, 0]
            y_coords = self.coords[0:self.test_point_batch_size][:, 1]
            model_float[x_coords, y_coords] = np.reshape((model_out >self.sampling_threshold).astype(np.uint8),[self.test_point_batch_size])  # self.sampling_threshold
            plt.subplot(3,4,i+1)
            plt.axis('off')
            plt.imshow((model_float>0)*255)

        plt.show()



    def get_z(self, config):
        # load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        pt_path = self.checkpoint_dir + '/' + self.model_dir + '/' + self.part_name + '_train_z.pt'
        shape_num = len(self.data_ims)


        self.im_network.eval()
        print(shape_num)
        z_zero=np.zeros([1,512],np.float32)
        for t in range(shape_num):
            batch_ims = self.data_ims[t:t + 1].astype(np.float32)
            batch_ims = torch.from_numpy(batch_ims)
            batch_ims = batch_ims.to(self.device)
            out_z, _ = self.im_network(batch_ims, None, None, is_training=False)
            z_zero=np.vstack((z_zero, out_z.detach().cpu().numpy()))

        z_zero=z_zero[1:,:]
        torch.save(pt_path,z_zero)
        print("[z]")



if __name__ == "__main__":
    from data_processing.data_utils import load_psi_data
    import os
    from matplotlib import pyplot as plt

    BASE_DIR =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cate = 'airplane'
    seen_split = 'train'
    use_cuda = torch.cuda.is_available()
    num_parts = 4
    batch_size = 1
    root = os.path.join(BASE_DIR, 'data/sketch_out')
    normalized_scale =128# {'chair_arm': 160, 'chair_back': 128, 'chair_leg': 128, 'chair_seat': 160}
    # parts = {'airplane': ['airplane_body', 'airplane_wing', 'airplane_tail', 'airplane_engine'],
    #          'chair': ['chair_arm', 'chair_back', 'chair_leg', 'chair_seat'], }

    ##todo: ===================================================================================================
    train_aes_loaders = load_psi_data(os.path.join(BASE_DIR, 'data/sketch_out'), cate, seen_split, num_parts=4,
                                      normalized_scale=normalized_scale,
                                      batch_size=3, disorderKey=True, _sk_representation='')





    #
