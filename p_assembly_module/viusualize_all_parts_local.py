from data_processing.data_utils import load_aen_data,load_pcn_data_local
import os
import torch
from models import parts_aes_graphs,get_model_pcn,get_model_pcn_single

import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from torch.autograd import Variable
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
from skimage.morphology import medial_axis, skeletonize
from utils.vector_resize import generate_single

parser = argparse.ArgumentParser(description='partAssembly')
parser.add_argument('--Maxepoch_ae', type=int, default=201)
parser.add_argument('--Maxepoch_pc', type=int, default=351)
parser.add_argument('--cate', type=str, default='chair')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--batch_size_pcn', type=int, default=20)
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--root', type=str, default='')
parser.add_argument('--num_parts', type=int, default=4)


parser.add_argument('--trans_flag',  action='store_true', default=True)
parser.add_argument('--train_display_interval', type=int, default=20)
parser.add_argument('--ae_lr', type=float, default=0.002)
parser.add_argument('--pc_lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr_policy', type=str, default='step')
parser.add_argument('--lr-decay-iters', type=int, default=30,
                                 help='multiply by a gamma every lr_decay_iters iterations')

opt = parser.parse_args()



cate_parts={'chair':4,'vase':4,'lampa':3,'lampc':3,'mug':2,'monitor':3,'airplane':4,'car':3,'guitar':3,'knife':2,'table':3}


pc_lr_cate={'chair':0.0001,'vase':0.00015,'lampa':0.00009,'lampc':0.0001,'mug':0.00015,'monitor':0.0001,'airplane':0.0001,'car':0.0001,'guitar':0.0001,'knife':0.0001,'table':0.0001}
batch_pc_cate={'chair':64,'vase':20,'lampa':20,'lampc':20,'mug':20,'monitor':64,'airplane':64,'car':64,'guitar':40,'knife':64,'table':32}
epoch_pc_cate={'chair':351,'vase':351,'lampa':401,'lampc':401,'mug':351,'monitor':351,'airplane':600,'car':351,'guitar':351,'knife':351,'table':351}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR='../../'
root = os.path.join(BASE_DIR, 'data/sketch_out')
opt.root=root
opt.pc_lr=pc_lr_cate[opt.cate]
opt.num_parts=cate_parts[opt.cate]

opt.cate='airplane'
opt.root=root
opt.num_parts=cate_parts[opt.cate]
use_cuda = torch.cuda.is_available()

device=torch.device('cuda:0')#.format(gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

model_dir='../../trained_models'

def train(opt):
    ii = 0
    start_epoch = 0


    # ## initialize the ae datalaoder
    # ae_dataloader = load_aen_data(data_path=opt.root, category=opt.cate, phase='train', num_parts=opt.num_parts, batch_size=opt.batch_size,disorderKey=True)

    ## initialize the aen
    aen = parts_aes_graphs(opt.num_parts)



    if use_cuda:
        for i in range(opt.num_parts):
                aen[i] = aen[i].encoder.cuda()
    ## reload the weights of the aen
    # aen_pt_dir='aen.pt'
    aen_pt_dir = os.path.join(BASE_DIR, model_dir, opt.cate+'_aen_local.pt')
    check_point = torch.load(aen_pt_dir)
    for partId in range(opt.num_parts):
        aen[partId].load_state_dict(check_point[partId]['net'])

    # ## initialize the pc datalaoder
    pc_dataloader, num_parts = load_pcn_data_local(data_path=opt.root, category=opt.cate, phase='train', batch_size=opt.batch_size_pcn,
                                          disorderKey=False,trans_flag=opt.trans_flag)
    # ## train pc_model

    pcn = get_model_pcn(num_parts)

    if use_cuda:
        pcn=pcn.cuda()


    ## reload the weights of the aen
    pcn_pt_dir = os.path.join(BASE_DIR,model_dir, opt.cate+ '_pcn_local.pt')
    check_point = torch.load(pcn_pt_dir)

    pcn.trans.load_state_dict(check_point['trans'])
    with torch.no_grad():
        for i, data in enumerate(pc_dataloader):
            batch_c = data[0][0].size()[0]
            x = torch.zeros((opt.num_parts, batch_c, 256, 256))
            x_b = torch.zeros((batch_c, opt.num_parts, 256, 256))
            y = torch.zeros((batch_c, opt.num_parts, 256, 256))
            y_b = torch.zeros((batch_c, opt.num_parts, 256, 256))
            gt_mask = torch.zeros((batch_c, opt.num_parts, 1))
            for p in range(opt.num_parts):
                for j in range(batch_c):
                    gts = data[0]
                    trans = data[1]

                    # im_s, im_s_b, im_n, im_n_b, is_full = data[p]
                    im_s = gts[p][:, :, :, 0]
                    im_s_b = gts[p][:, :, :, 1]
                    im_t = trans[p][:, :, :, 0]
                    im_t_b = trans[p][:, :, :, 1]

                    x[p, j, ...] = im_t[j].squeeze()
                    y[j, p, ...] = im_s[j].squeeze()
                    x_b[j, p, ...] = im_t_b[j].squeeze()
                    y_b[j, p, ...] = im_s_b[j].squeeze()

            # num_parts_flag = torch.mean(gt_mask, 2)

            p_enc_codes = []
            p_gts = []
            for p in range(opt.num_parts):
                input_enc = x[p, :, :, :].cuda().unsqueeze(1).detach()
                code, p_gt = aen[p](input_enc)  # 1,512
                p_enc_codes.append(code.detach())
                p_gts.append(p_gt.detach())

            p_enc_codes = torch.cat(p_enc_codes, axis=-1)

            p_gt_ = torch.cat(p_gts, axis=1)

            p_enc_codes, p_gt_ = p_enc_codes.cuda(), p_gt_.cuda()
            x_b = x_b.to(device).requires_grad_(True)

            y = y.cuda().detach()
            y_b = y_b.cuda()

            y_hat, y_b_hat, theta_ = pcn(p_enc_codes, p_gt_, x_b)

            # for n_time in range(2):
            #     ## re initalization
            #     x=y_hat.transpose(0,1)
            #     x_b= y_b_hat
            #     p_enc_codes = []
            #     p_gts = []
            #     for p in range(opt.num_parts):
            #         input_enc = x[p, :, :, :].cuda().unsqueeze(1).detach()
            #         code, p_gt = aen[p](input_enc)  # 1,512
            #         p_enc_codes.append(code.detach())
            #         p_gts.append(p_gt.detach())
            #
            #     p_enc_codes = torch.cat(p_enc_codes, axis=-1)
            #     p_gt_ = torch.cat(p_gts, axis=1)
            #
            #     p_enc_codes, p_gt_ = p_enc_codes.cuda(), p_gt_.cuda()
            #     x_b = x_b.to(device).requires_grad_(True)
            #
            #     y_hat, y_b_hat, theta_ = pcn(p_enc_codes, p_gt_, x_b)



            for coun in range(batch_c):
                m, n = opt.num_parts, 5
                plt.figure(1,figsize=(10,8))
                for parts_id in range(m):
                    plt.subplot(m, n, n * parts_id + 1)
                    plt.axis('off')
                    plt.title('input')
                    plt.imshow(p_gt_[coun, parts_id].detach().cpu().squeeze().numpy())

                    plt.subplot(m, n, n * parts_id + 2)
                    plt.axis('off')
                    plt.title('warped')
                    plt.imshow(y_hat[coun,parts_id].detach().cpu().squeeze().numpy())
                    plt.subplot(m, n, n * parts_id + 3)
                    plt.axis('off')
                    plt.title('gt')
                    plt.imshow(y[coun,parts_id].detach().cpu().squeeze().numpy())
                    plt.subplot(m, n, n * parts_id + 4)
                    plt.axis('off')
                    plt.title('warped_bbx')
                    plt.imshow(y_b_hat[coun,parts_id].detach().cpu().squeeze().numpy())
                    plt.subplot(m, n, n * parts_id + 5)
                    plt.axis('off')
                    plt.title('gt_bbx')
                    plt.tight_layout()
                    plt.imshow(y_b[coun,parts_id].detach().cpu().squeeze().numpy())

                plt.figure(2)
                strokes = np.zeros((256, 256), dtype="uint8")
                for parts_id in range(m):
                    part = y[coun, parts_id].detach().cpu().squeeze().numpy()
                    strokes[part > 0] = 1
                plt.subplot(133)

                plt.axis('off')
                plt.title('ground_truth')
                plt.tight_layout()
                plt.imshow(strokes)

                strokes_ = np.zeros((256, 256), dtype="uint8")
                for parts_id in range(m):
                    part = y_hat[coun, parts_id].detach().cpu().squeeze().numpy()
                    strokes_[part > 0] = 1
                plt.subplot(132)
                plt.axis('off')
                plt.title('assembly_sketch')
                plt.tight_layout()
                plt.imshow(generate_single(strokes_))

                strokes_ = np.zeros((256, 256), dtype="uint8")
                for parts_id in range(m):
                    part = p_gt_[coun, parts_id].detach().cpu().squeeze().numpy()
                    strokes_[part > 0] = 1
                plt.subplot(131)
                plt.axis('off')
                plt.title('inputs')
                plt.tight_layout()
                plt.imshow(generate_single(strokes_))

                plt.show()
                print()



if __name__ == '__main__':
    train(opt)
