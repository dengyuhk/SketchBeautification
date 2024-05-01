from data_processing.data_utils import load_pcn_data_local,load_aen_data
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import time
import numpy as np
from models import parts_aes_graphs,get_model_pcn,get_model_pcn_single
import argparse
from framework import get_scheduler,update_learning_rate,save_ae_model,save_ae_model_joint,save_pc_model,pc_one_epoch_local,save_pc_ae_model
from framework import ae_one_epoch,pc_one_epoch#,pc_one_epoch_single
from skimage.morphology import medial_axis, skeletonize
from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser(description='partAssembly')
parser.add_argument('--Maxepoch_ae', type=int, default=201)
parser.add_argument('--Maxepoch_pc', type=int, default=800)
parser.add_argument('--cate', type=str, default='chair')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--batch_size_pcn', type=int, default=20)
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--root', type=str, default='')
parser.add_argument('--num_parts', type=int, default=4)

parser.add_argument('--trans_flag',  action='store_true', default=True)
parser.add_argument('--train_display_interval', type=int, default=2)
parser.add_argument('--ae_lr', type=float, default=0.002)
parser.add_argument('--pc_lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr_policy', type=str, default='step')
parser.add_argument('--lr-decay-iters', type=int, default=30,
                                 help='multiply by a gamma every lr_decay_iters iterations')

opt = parser.parse_args()



cate_parts={'chair':4,'vase':4,'lampa':3,'lampc':3,'mug':2,'monitor':3,'airplane':4,'car':3,'guitar':3,'knife':2,'table':3}

opt.cate='vase'#'knife'#'vase'
pc_lr_cate={'chair':0.0001,'vase':0.0001,'lampa':0.0001,'lampc':0.0001,'mug':0.0001,'monitor':0.0001,'airplane':0.0001,'car':0.0001,'guitar':0.0001,'table':0.0001}
batch_pc_cate={'chair':64, 'vase':16, 'lampa':16, 'lampc':16, 'mug':16, 'monitor':16, 'airplane':16, 'car':16, 'guitar':16, 'table':16}
epoch_pc_cate={'chair':601,'vase':601,'lampa':601,'lampc':601,'mug':601,'monitor':601,'airplane':601,'car':601,'guitar':601,'table':601}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR='../../'#'
root = os.path.join(BASE_DIR, 'data/sketch_out')
save_model_dir='tranied_model'
if not os.path.exists(save_model_dir): os.makedirs(save_model_dir)
opt.root=root
opt.pc_lr=pc_lr_cate[opt.cate]
opt.num_parts=cate_parts[opt.cate]
use_cuda = torch.cuda.is_available()
writer = SummaryWriter(logdir=os.path.join("runs",'baseline_'+opt.cate))

def train(opt):
    ii = 0
    start_epoch = 0
    opt.pc_lr = pc_lr_cate[opt.cate]
    opt.num_parts = cate_parts[opt.cate]
    opt.Maxepoch_pc = epoch_pc_cate[opt.cate]
    opt.batch_size_pcn = batch_pc_cate[opt.cate]


    ## initialize the aen
    aen = parts_aes_graphs(opt.num_parts)

    if use_cuda:
        for i in range(opt.num_parts):
                aen[i] = aen[i].encoder.cuda()

    # print params of aen
    print('aen params:\t')
    for param_tensor in aen[0].state_dict():
        print(param_tensor, "\t", aen[0].state_dict()[param_tensor].size())
    print('\n')

    #initalize pc_model
    save_loss=1e20
    pcn = get_model_pcn(opt.num_parts)

    # print params of pcn
    print('pcn params:\t')
    for param_tensor in pcn.state_dict():
        print(param_tensor, "\t", pcn.state_dict()[param_tensor].size())
    print('\n')

    if use_cuda:
        pcn=pcn.cuda()
    opt.lr_decay_iters = 80

    ## ============================resume================================
    ##aen
    # resume_flage = True ## need to modify
    # if resume_flage:
    #     start_epoch=60 ## need to modify
    #     opt.pc_lr=0.0000500 ## need to modify
    #
    #     aen_pt_dir = os.path.join(BASE_DIR, pretrained_model_dir, opt.cate + '_aen_local.pt')
    #     check_point = torch.load(aen_pt_dir)
    #     for partId in range(opt.num_parts):
            # aen[partId].load_state_dict(check_point[partId]['net'])

        # pcn_pt_dir = os.path.join(BASE_DIR, pretrained_model_dir, opt.cate + '_pcn_local.pt')
        # check_point = torch.load(pcn_pt_dir)
        # pcn.trans.load_state_dict(check_point['trans'])
    # else:
    #     pass


    params = list(aen[0].parameters())
    for partId in range(1,opt.num_parts):
        params= params+ list(aen[partId].parameters())
    params= params + list(pcn.parameters())

    pc_optimizer = torch.optim.Adam(params, lr=opt.pc_lr, betas=(opt.beta1, 0.999))
    pc_scheduler = get_scheduler(pc_optimizer, opt)

    # ## initialize the pc datalaoder
    pc_dataloader, num_parts = load_pcn_data_local(data_path=opt.root, category=opt.cate, phase='train', batch_size=opt.batch_size_pcn,
                                          disorderKey=True,trans_flag=opt.trans_flag)
    ## seems dataloader doesn't work, here we rewrite it
    # data_pt_name = os.path.join(root, '{}_{}_{}_local.pt'.format(opt.cate, 'sketch', 'pc'))
    # if os.path.exists(data_pt_name):
    #     print('loading training data of {}'.format(opt.cate))
    #     data_dict = torch.load(data_pt_name)
    #     data_dict = data_dict['train']
    #     data_imgs=data_dict['imgs']
    # else:
    #     print("error: cannot load " + data_pt_name)
    #     exit(0)
    #
    # shape_num=len(data_dict['imgs'])
    # batch_index_list = np.arange(shape_num)
    #
    # print("\n\n----------net summary----------")
    # print("training samples   ", shape_num)
    # print("-------------------------------\n\n")
    #
    # start_time = time.time()
    # batch_num = int(shape_num / opt.batch_size)


    for epoch in range(start_epoch,opt.Maxepoch_pc):
        ## define the single layer
        # loss_c=pc_one_epoch(pcn, aen, pc_dataloader, opt, epoch, pc_optimizer)#,writer)
        ## seems dataloader cannot work
        # np.random.shuffle(batch_index_list)
        loss_c = pc_one_epoch_local(pcn, aen,pc_dataloader, opt, epoch, pc_optimizer, writer)
        # loss_c = pc_one_epoch_local(pcn, aen, pc_dataloader, opt, epoch, pc_optimizer)#, writer)
        # loss_c = pc_one_epoch_local(pcn, aen, data_imgs,batch_index_list,batch_num, opt, epoch, pc_optimizer ,writer)
        # loss_c = pc_one_epoch_single(pcn, aen[0], pc_dataloader, opt, epoch, pc_optimizer,writer)
        if epoch==opt.Maxepoch_pc-1:
            save_pc_ae_model(aen, pcn, save_model_dir+'/'+opt.cate + '_aen_local_final.pt', save_model_dir+'/'+opt.cate + '_pcn_local_final.pt', opt)
            save_loss = loss_c
        else:
            if loss_c < save_loss:
                save_pc_ae_model(aen, pcn,  save_model_dir+'/'+opt.cate+'_aen_local.pt',save_model_dir+'/'+opt.cate+'_pcn_local.pt',opt)
                save_loss=loss_c

        update_learning_rate(pc_optimizer, pc_scheduler)
    writer.close()



if __name__ == '__main__':
    train(opt)
