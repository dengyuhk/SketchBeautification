import numpy as np
import torch
import torch.nn as nn
# from data_processing.data_utils import load_aen_data#,get_pcn_batch
import os
from matplotlib import pyplot as plt
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    elif isinstance(m,nn.ConvTranspose2d):
        # nn.init.orthogonal_(m.weight.data, gain=0.6)
        nn.init.xavier_uniform_(m.weight)
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    elif isinstance(m, torch.nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias,0)
    return

## todo:===================================for part Auto_encoder
class ae_encoder(nn.Module):
    def __init__(self, kernelSize=4, norm=nn.InstanceNorm2d, activation=nn.LeakyReLU(True)):
        super(ae_encoder, self).__init__()
        self.kernelSize = kernelSize
        self.norm = norm
        self.activation = activation

        model=[
            nn.Conv2d(1, 32, kernel_size=self.kernelSize, stride=2, padding=1, padding_mode='replicate',bias=False),  # b, 32, 128, 128
            self.norm(32),
            self.activation,
            nn.Conv2d(32, 64, kernel_size=self.kernelSize, stride=2, padding=1, padding_mode='replicate',bias=False),  # b, 32, 64, 64
            self.norm(64),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=self.kernelSize, stride=2, padding=1, padding_mode='replicate',bias=False),  # b, 64, 32, 32
            self.norm(128),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=self.kernelSize, stride=2, padding=1, padding_mode='replicate',bias=False),  # b, 64, 16, 16
            self.norm(256),
            self.activation,
            nn.Conv2d(256, 512, kernel_size=self.kernelSize, stride=2, padding=1,padding_mode='replicate',bias=False),  # b, 128, 8, 8
            self.norm(512),
            self.activation,]

        self.model = nn.Sequential(*model)
        self.fc = nn.Sequential(nn.Linear(512 *8 * 8, 512))
        self.model.apply(weights_init)
        self.fc.apply(weights_init)
    def forward(self, x):
        feature_map = self.model(x)
        feature_map=feature_map.reshape(feature_map.size()[0],-1)
        code = self.fc(feature_map)
        return code,x

## todo: ================================for part Auto_decoder
class ae_decoder(nn.Module):
    def __init__(self, kernelSize=4, norm=nn.InstanceNorm2d, activation=nn.LeakyReLU(True)):
        super(ae_decoder, self).__init__()
        self.activation = activation  # nn.Tanh()#
        self.norm = norm
        self.kernelSize = kernelSize
        model =[
            nn.ConvTranspose2d(512, 256, kernel_size=self.kernelSize, stride=2, padding=1,bias=False),  # b, 16, 16, 16
            self.norm(256),
            self.activation,
            nn.ConvTranspose2d(256, 128, kernel_size=self.kernelSize, stride=2, padding=1,bias=False),  # b, 16, 32, 32
            self.norm(128),
            self.activation,
            nn.ConvTranspose2d(128, 64, kernel_size=self.kernelSize, stride=2, padding=1,bias=False),  # b, 16, 64, 64
            self.norm(64),
            self.activation,
            nn.ConvTranspose2d(64, 32, kernel_size=self.kernelSize, stride=2, padding=1,bias=False),  # b, 16, 128, 128
            self.norm(32),
            self.activation,
            nn.ConvTranspose2d(32, 1, kernel_size=self.kernelSize, stride=2, padding=1,bias=False),  # b, 1, 256, 256
            nn.ReLU(),
            ]

        self.model = nn.Sequential(*model)
        self.fc_t = nn.Sequential(nn.Linear(512, 512 * 8 * 8))
        self.model.apply(weights_init)
        self.fc_t.apply(weights_init)

    def forward(self, x):
        code= self.fc_t(x)
        feature_map=code.reshape(-1,512,8,8)
        im = self.model(feature_map)
        return im

class get_model_ae(nn.Module):
    def __init__(self,):
        super(get_model_ae, self).__init__()
        self.kernelSize = 4
        self.activation = nn.LeakyReLU(True)  # nn.Tanh()#
        self.norm = nn.InstanceNorm2d

        self.encoder = ae_encoder(self.kernelSize, self.norm, self.activation)
        self.decoder = ae_decoder(self.kernelSize, self.norm, self.activation)

    def forward(self,x):
        latent,im_ori = self.encoder(x)
        x = self.decoder(latent)
        return x

##todo: ===============================================================
def parts_aes_graphs(num_parts):# build_parts_aes_graphs
    ae_graph = []
    for i in range(num_parts):
        ae_graph.append(get_model_ae())
    return ae_graph

##todo: ============================================= For global assembly:
class get_model_pcn(nn.Module):
    def __init__(self,num_parts):
        super(get_model_pcn, self).__init__()
        self.num_parts= num_parts
        bias_initializer = np.array([[1, 0, 0] for _ in range(self.num_parts)])
        bias_initializer = bias_initializer.astype('float32').flatten()
        trans= [
           nn.Linear(self.num_parts*512,1024),
           nn.Linear(1024,256),
           nn.Linear(256, 128),
           nn.Linear(128,self.num_parts*3)
        ]
        ## 512*4 -> 1024 -> 256 ->128-> 4*6
        self.trans=nn.Sequential(*trans)
        self.trans[-1].weight.data.fill_(0)
        self.trans[-1].bias.data =torch.FloatTensor(bias_initializer)

    def forward(self,p_encs,x,x_b):
        ## obtain the transfromation matrix
        zeros_dims = torch.zeros([x.size()[0], 1]).cuda()
        trans = self.trans(p_encs)
        '''
        sx   0    tx
        0    sy   ty
        '''
        trans_theta = torch.cat((trans[:, 0].unsqueeze(1), zeros_dims, trans[:, 1].unsqueeze(1),
                                 zeros_dims,trans[:, 0].unsqueeze(1), trans[:, 2].unsqueeze(1)), axis=1)

        for p in range(1, self.num_parts):
            start_ind = 3 * p
            trans_theta = torch.cat((trans_theta,trans[:, start_ind+0].unsqueeze(1),zeros_dims,trans[:, start_ind+1].unsqueeze(1),
                                   zeros_dims,trans[:, start_ind+0].unsqueeze(1), trans[:, start_ind+2].unsqueeze(1)), axis=1)
        theta = trans_theta.view(-1, self.num_parts,2, 3)

        ##peform warpping transfromation
        x_warp=[]
        x_b_warp=[]
        for i in range(self.num_parts):
            grid = F.affine_grid(theta[:,i], x[:,i].unsqueeze(1).size())  ## output: [64,28,28,2]
            x_warp.append(F.grid_sample(x[:,i].unsqueeze(1), grid).detach())
            x_b_warp.append(F.grid_sample(x_b[:,i].unsqueeze(1), grid))
        y_hat = torch.cat(x_warp, axis=1)
        y_b_hat=torch.cat(x_b_warp,axis=1)
        return y_hat,y_b_hat, theta






class get_model_pcn_single(nn.Module):
    def __init__(self,):
        super(get_model_pcn_single, self).__init__()
        # self.num_parts= num_parts
        # bias_initializer = np.array([[1, 0, 0, 0, 1, 0] for _ in range(self.num_parts)])
        bias_initializer = np.array([1,0,0])
        bias_initializer = bias_initializer.astype('float32').flatten()
        trans= [
           # nn.Linear(self.num_parts*512,1024),
           nn.Linear(512, 256),
           # nn.Linear(1024,256),
           nn.Linear(256, 128),
           nn.Linear(128,3)
           # nn.Linear(128, 6)
        ]
        ## 512*4 -> 1024 -> 256 ->128-> 4*6
        self.trans=nn.Sequential(*trans)
        # self.trans.apply(weights_init)

        self.trans[-1].weight.data.fill_(0)
        self.trans[-1].bias.data =torch.FloatTensor(bias_initializer)

    def forward(self,p_encs,x,x_b):
        ## obtain the transfromation matrix
        zeros_dims = torch.zeros([x.size()[0], 1]).cuda()

        trans = self.trans(p_encs)
        # a=trans[:0]
        # b=trans[:,0].unsqueeze(1)
        '''
        sx   0    tx
        0    sx   ty
        '''
        trans_theta=torch.cat((trans[:,0].unsqueeze(1),zeros_dims,trans[:,1].unsqueeze(1),zeros_dims,trans[:,0].unsqueeze(1),trans[:,2].unsqueeze(1)),axis=1)
        # trans_theta = torch.cat((trans[:, 0].unsqueeze(1), zeros_dims, zeros_dims,
        #                          zeros_dims, trans[:, 0].unsqueeze(1), zeros_dims), axis=1)

        # trans_mat = tf.concat(
        #     (tf.expand_dims(trans[:, 0], axis=1), zeros_col, zeros_col, tf.expand_dims(trans[:, 1], axis=1),
        #      zeros_col, tf.expand_dims(trans[:, 2], axis=1), zeros_col, tf.expand_dims(trans[:, 3], axis=1),
        #      zeros_col, zeros_col, tf.expand_dims(trans[:, 4], axis=1), tf.expand_dims(trans[:, 5], axis=1),
        #
        #      tf.expand_dims(trans[:, 6], axis=1), zeros_col, zeros_col, tf.expand_dims(trans[:, 7], axis=1),
        #      zeros_col, tf.expand_dims(trans[:, 8], axis=1), zeros_col, tf.expand_dims(trans[:, 9], axis=1),
        #      zeros_col, zeros_col, tf.expand_dims(trans[:, 10], axis=1), tf.expand_dims(trans[:, 11], axis=1)), axis=1)
        # for p in range(2, num_parts):
        #     start_ind = 6 * p
        #     trans_mat = tf.concat((trans_mat, tf.expand_dims(trans[:, start_ind], axis=1), zeros_col, zeros_col,
        #                            tf.expand_dims(trans[:, start_ind + 1], axis=1),
        #                            zeros_col, tf.expand_dims(trans[:, start_ind + 2], axis=1), zeros_col,
        #                            tf.expand_dims(trans[:, start_ind + 3], axis=1),
        #                            zeros_col, zeros_col, tf.expand_dims(trans[:, start_ind + 4], axis=1),
        #                            tf.expand_dims(trans[:, start_ind + 5], axis=1)), axis=1)
        # trans = trans_mat.view(-1,2, 2)
        theta = trans_theta.view(-1, 2, 3)
        ##peform warpping transfromation


        # print(theta[0])
        # grid = F.affine_grid(theta[:, i, :, :], x[:, i, :, :].unsqueeze(1).size())
        grid = F.affine_grid(theta, x.size())
        y_hat = F.grid_sample(x, grid)
        y_b_hat = F.grid_sample(x_b, grid)
        # x_warp=[]
        # for i in range(self.num_parts):
        #     grid = F.affine_grid(theta[:,i,:,:], x[:,i,:,:].unsqueeze(1).size())  ## output: [64,28,28,2]
        #     x_warp.append(F.grid_sample(x[:,i,:,:].unsqueeze(1), grid))
        #

        return y_hat,y_b_hat,trans



if __name__ == '__main__':

    BASE_DIR = '../../'#os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cate = 'chair'
    seen_split = 'train'
    num_parts = 4
    batch_size = 1
    use_cuda = torch.cuda.is_available()
    root = os.path.join(BASE_DIR, 'data/sketch_out')
    ##todo:====================================================================
    ## test the pae_net
    # dataloader = load_aen_data(data_path=root, category=cate, phase='train', num_parts=num_parts, batch_size=batch_size,
    #                            disorderKey=True)
    # ##==test the single pae_net
    # # get_model_ae=ae_encoder()
    # ##==test the full pae
    # # model= get_model_ae(num_parts)
    # ##==test the pae_graph
    # model = parts_aes_graphs(num_parts)
    # if use_cuda:
    #     for i in range(num_parts):
    #         model.ae_graph[i] = model.ae_graph[i].cuda()
    # ## for train_aen_loader in dataloader:
    # part_index=0
    # for i, d in enumerate(dataloader[part_index]):
    #     data = d.cuda().unsqueeze(1)
    #     output = model.ae_graph[part_index](data)
    #     plt.imshow(output.detach().cpu().numpy().squeeze())
    #     plt.show()




    ##todo:====================================================================
    ## test the pcn_net
    # dataloader,num_parts = load_pcn_data(data_path=root, category=cate,phase='train',batch_size=batch_size,disorderKey=True)
    # x = torch.zeros((num_parts, batch_size, 256, 256))
    # y = torch.zeros((batch_size, num_parts, 256, 256))
    # y_mask = torch.zeros((batch_size, num_parts, 256, 256))
    #
    # for i, data in enumerate(dataloader):
    #     point_sets=data
    #     for p in range(num_parts):
    #         for j in range(batch_size):
    #             im_s, im_n, is_full = point_sets[p]
    #             x[p, j, ...] = im_n[j].squeeze()
    #             y[j, p, ...] = im_s[j].squeeze()
    #             y_mask[j, p] = is_full[j]
    #     break
    # graph_aes = parts_aes_graphs(num_parts)
    # if use_cuda:
    #     for i in range(num_parts):
    #         graph_aes[i] = graph_aes[i].cuda()
    # p_enc_codes=[]
    # p_gts=[]
    # for p in range(num_parts):
    #     input_enc=x[p,:,:,:].cuda().unsqueeze(1)
    #     code,p_gt=graph_aes[p].encoder(input_enc) # 1,512
    #     p_enc_codes.append(code)
    #     p_gts.append(p_gt)#
    #
    # p_enc_codes=torch.cat(p_enc_codes,axis=-1)
    # p_gts=torch.cat(p_gts,axis=1)
    #
    # model_pcn = get_model_pcn(num_parts)
    # if use_cuda:
    #     model_pcn.cuda()
    # y_hat=model_pcn(p_enc_codes,p_gts)
    #
    # m,n=4,3
    # for parts_id in range(num_parts):
    #     plt.subplot(m, n, 3 * parts_id + 1)
    #     plt.title('input')
    #     plt.imshow(p_gts[0][parts_id].detach().cpu().numpy())
    #     plt.subplot(m, n, 3 * parts_id + 2)
    #     plt.title('warped')
    #     plt.imshow(y_hat[0][parts_id].detach().cpu().numpy())
    #     plt.subplot(m, n, 3 * parts_id + 3)
    #     plt.title('gt')
    #     plt.imshow(y[0][parts_id].detach().cpu().numpy())
    # plt.show()
