import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
import os
import torchvision
import torch.nn.functional as F

device = torch.device('cuda:0')  # .format(gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
aen_loss_func = torch.nn.MSELoss().to(device)  # L2Loss
pcn_loss_func = torch.nn.L1Loss().to(device)


def update_learning_rate(optimizer, scheduler):
    """
    update learning rate (called once every epoch)
    """
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """

    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.long().squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = logits  # torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        pass
        # true_1_hot = torch.eye(num_classes)[true.long().squeeze(1)]
        # true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        # probas=logits
        # probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def ae_one_epoch(model, data_loader, opt, epoch, optimizer, writer):
    """
    feed data nad network: model, training_part_data
    """
    loss_sum_avg = []
    for i, data in enumerate(data_loader):

        data = data.cuda().unsqueeze(1)
        input_ = data.to(device).requires_grad_(True)
        optimizer.zero_grad()
        out = model(input_)
        loss = aen_loss_func(out, input_) * 10
        loss.backward()
        loss_sum_avg.append(loss.item())
        optimizer.step()

        writer.add_image('x', torchvision.utils.make_grid(out[0:4]), i)
        writer.add_image('y_hat', torchvision.utils.make_grid(input_[0:4]), i)
        # writer.add_image('gt', torchvision.utils.make_grid(y)[0:8], i)
        if i % opt.train_display_interval == 0:
            print('Train Epoch: {}  iters:{} Loss: {:.6f} '.format(epoch, i, loss.item()))
    print('Train Epoch: {} \tAvg Loss: {:.6f}'.format(epoch, sum(loss_sum_avg) / len(loss_sum_avg)))

    return sum(loss_sum_avg) / len(loss_sum_avg)


def pc_one_epoch(model, ae_model, data_loader, opt, epoch, optimizer):  # ,_writer):
    """
    feed data nad network: model, training_part_data
    """
    loss_sum = []
    # try:
    for i, data in enumerate(data_loader):
        point_sets = data
        batch_c = data[0][0].size()[0]
        x = torch.zeros((opt.num_parts, batch_c, 256, 256))
        x_b = torch.zeros((batch_c, opt.num_parts, 256, 256))
        y = torch.zeros((batch_c, opt.num_parts, 256, 256))
        y_b = torch.zeros((batch_c, opt.num_parts, 256, 256))
        gt_mask = torch.zeros((batch_c, opt.num_parts, 1))
        # id_theta=
        for p in range(opt.num_parts):
            for j in range(batch_c):
                im_s, im_s_b, im_n, im_n_b, is_full = point_sets[p]
                x[p, j, ...] = im_n[j].squeeze()
                y[j, p, ...] = im_s[j].squeeze()
                x_b[j, p, ...] = im_n_b[j].squeeze()
                y_b[j, p, ...] = im_s_b[j].squeeze()
                gt_mask[j, p] = is_full[j] * 1
        num_parts_flag = torch.mean(gt_mask, 2)

        p_enc_codes = []
        p_gts = []
        for p in range(opt.num_parts):
            input_enc = x[p, :, :, :].cuda().unsqueeze(1)
            code, p_gt = ae_model[p].encoder(input_enc)  # 1,512
            p_enc_codes.append(code)
            p_gts.append(p_gt)

        p_enc_codes = torch.cat(p_enc_codes, axis=-1)

        p_gt_ = torch.cat(p_gts, axis=1)

        # gt_masks=gt_mask.cuda()
        p_enc_codes, p_gt_ = p_enc_codes.cuda(), p_gt_.cuda()
        x_b = x_b.to(device).requires_grad_(True)

        y = y.cuda()
        y_b = y_b.cuda()
        optimizer.zero_grad()
        y_hat, y_b_hat, theta_ = model(p_enc_codes, p_gt_, x_b)

        bl1loss = 0
        biouloss = 0
        imloss = 0
        for p in range(opt.num_parts):
            y_b_ = y_b[:, p].unsqueeze(1)  # *num_parts_flag[:,p].unsqueeze(1)
            y_b_hat_ = y_b_hat[:, p].unsqueeze(1)  # *num_parts_flag[:,p].unsqueeze(1)
            biouloss += jaccard_loss(y_b_, y_b_hat_)
            bl1loss += pcn_loss_func(y_b_, y_b_hat_)
            # imloss += 0.001 * pcn_loss_func(y[:,p].unsqueeze(1), y_hat[:,p].unsqueeze(1))

        num_parts_batch = torch.sum(num_parts_flag)
        bloss_mean = bl1loss / num_parts_batch
        # bloss_mean = biouloss / num_parts_batch
        # imloss_mean =imloss/num_parts_batch
        loss_ = bloss_mean  # +imloss_mean

        loss_.backward()
        loss = biouloss / num_parts_batch

        loss_sum.append(loss.item())
        optimizer.step()

        # _writer.add_image('y_hat', torchvision.utils.make_grid(y_hat[0,:].unsqueeze(1)), i)
        # _writer.add_image('gt', torchvision.utils.make_grid(y[0,:].unsqueeze(1)), i)
        # _writer.add_image('y_b_hat', torchvision.utils.make_grid(y_b_hat[0,:].unsqueeze(1)), i)
        # _writer.add_image('box_gt', torchvision.utils.make_grid(y_b[0,:].unsqueeze(1)), i)

        if i % opt.train_display_interval == 0:
            print('Train Epoch: {} iters:{} Loss: {:.6f}'.format(epoch, i, loss.item()))

        torch.cuda.empty_cache()
        del y_hat, y, y_b_hat_, x_b, p_gt_
    print('Train Epoch: {} \tAvg Loss: {:.6f}'.format(epoch, sum(loss_sum) / len(loss_sum)))

    return sum(loss_sum) / len(loss_sum)


def pc_one_epoch_local(model, ae_model, data_loader, opt, epoch, optimizer, _writer):
    """
    feed data nad network: model, training_part_data
    """
    loss_sum = []
    # try:
    for i, data in enumerate(data_loader):

        batch_c = data[0][0].size()[0]
        x = torch.zeros((opt.num_parts, batch_c, 256, 256))
        x_b = torch.zeros((batch_c, opt.num_parts, 256, 256))
        y = torch.zeros((batch_c, opt.num_parts, 256, 256))
        y_b = torch.zeros((batch_c, opt.num_parts, 256, 256))
        gt_mask = torch.zeros((batch_c, opt.num_parts, 1))
        id_theta_e = torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0)
        id_theta_parts = id_theta_e.repeat(opt.num_parts, 1, 1).unsqueeze(0)
        id_theta_parts_batch = id_theta_parts.repeat(batch_c, 1, 1, 1)

        for p in range(opt.num_parts):
            for j in range(batch_c):
                gts = data[0]
                trans = data[1]
                flags = data[2]

                im_s = gts[p][:, :, :, 0]
                im_s_b = gts[p][:, :, :, 1]
                im_t = trans[p][:, :, :, 0]
                im_t_b = trans[p][:, :, :, 1]
                is_full = flags[p]

                ## affined sk and boundingbox
                x[p, j, ...] = im_t[j].squeeze()
                x_b[j, p, ...] = im_t_b[j].squeeze()

                ### gt sk and boundingbox
                y[j, p, ...] = im_s[j].squeeze()
                y_b[j, p, ...] = im_s_b[j].squeeze()

                gt_mask[j, p] = is_full[j] * 1

        num_parts_flag = torch.mean(gt_mask, 2)

        p_enc_codes = []
        p_inputs = []
        for p in range(opt.num_parts):
            input_enc = x[p, :, :, :].cuda().unsqueeze(1)
            code, p_input = ae_model[p](input_enc)  # [1,512]
            p_enc_codes.append(code)
            p_inputs.append(p_input)

        p_enc_codes = torch.cat(p_enc_codes, axis=-1)

        p_input_ = torch.cat(p_inputs, axis=1)

        # gt_masks=gt_mask.cuda()
        p_enc_codes, p_input_ = p_enc_codes.cuda(), p_input_.cuda()
        x_b = x_b.cuda()

        y = y.cuda()
        y_b = y_b.cuda()
        id_theta_parts_batch = id_theta_parts_batch.float().cuda()

        optimizer.zero_grad()
        y_hat, y_b_hat, theta_ = model(p_enc_codes, p_input_, x_b)

        bl1loss = 0
        biouloss = 0
        skloss = 0
        theta_loss = 0

        for p in range(opt.num_parts):
            y_b_ = y_b[:, p].unsqueeze(1)  # *num_parts_flag[:,p].unsqueeze(1)
            y_b_hat_ = y_b_hat[:, p].unsqueeze(1)  # *num_parts_flag[:,p].unsqueeze(1)
            biouloss += jaccard_loss(y_b_, y_b_hat_)
            bl1loss += pcn_loss_func(y_b_, y_b_hat_)
            skloss += pcn_loss_func(y[:, p].unsqueeze(1), y_hat[:, p].unsqueeze(1))
            theta_loss += aen_loss_func(theta_, id_theta_parts_batch)

        num_parts_batch = torch.sum(num_parts_flag)

        ##mask_l1_loss
        bl1loss_mean = bl1loss / num_parts_batch
        ##mask_iou_loss
        biouloss_mean = biouloss / num_parts_batch
        ##sk_loss
        skloss_mean = skloss / num_parts_batch
        ##affine_loss
        theta_loss_mean = theta_loss / num_parts_batch
        ##final loss
        loss_ = 100 * bl1loss_mean + 1 * skloss_mean + 1 * theta_loss_mean

        loss_.backward()
        optimizer.step()

        _writer.add_image('input', torchvision.utils.make_grid(p_input_[0, :].unsqueeze(1)), i)
        _writer.add_image('y_hat', torchvision.utils.make_grid(y_hat[0, :].unsqueeze(1)), i)
        _writer.add_image('gt', torchvision.utils.make_grid(y[0, :].unsqueeze(1)), i)
        _writer.add_image('y__b_hat', torchvision.utils.make_grid(y_b_hat[0, :].unsqueeze(1)), i)
        _writer.add_image('box_gt', torchvision.utils.make_grid(y_b[0, :].unsqueeze(1)), i)

        iou_loss_display = 100 * biouloss_mean  # biouloss / num_parts_batch
        if i % opt.train_display_interval == 0:
            print('Train Epoch: {} iters:{} Loss: {:.6f}'.format(epoch, i, iou_loss_display.item()))
        torch.cuda.empty_cache()
        del y_hat, y, y_b_hat_, x_b, p_input_
        loss_sum.append(iou_loss_display.item())

    print('Train Epoch: {} \tAvg Loss: {:.6f}'.format(epoch, sum(loss_sum) / len(loss_sum)))
    _writer.add_scalar('miou_sk__affine_loss', sum(loss_sum) / len(loss_sum), epoch)

    return sum(loss_sum) / len(loss_sum)


def pc_one_epoch_local_149(model, ae_model, data_loader, opt, epoch, optimizer):  # ,_writer):
    """
    feed data nad network: model, training_part_data
    """
    loss_sum = []
    # try:
    for i, data in enumerate(data_loader):

        batch_c = data[0][0].size()[0]
        x = torch.zeros((opt.num_parts, batch_c, 256, 256))
        x_b = torch.zeros((batch_c, opt.num_parts, 256, 256))
        y = torch.zeros((batch_c, opt.num_parts, 256, 256))
        y_b = torch.zeros((batch_c, opt.num_parts, 256, 256))
        gt_mask = torch.zeros((batch_c, opt.num_parts, 1))
        id_theta_e = torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0)
        id_theta_parts = id_theta_e.repeat(opt.num_parts, 1, 1).unsqueeze(0)
        id_theta_parts_batch = id_theta_parts.repeat(batch_c, 1, 1, 1)

        for p in range(opt.num_parts):
            for j in range(batch_c):
                gts = data[0]
                trans = data[1]
                flags = data[2]

                im_s = gts[p][:, :, :, 0]
                im_s_b = gts[p][:, :, :, 1]
                im_t = trans[p][:, :, :, 0]
                im_t_b = trans[p][:, :, :, 1]
                is_full = flags[p]

                ## affined sk and boundingbox
                x[p, j, ...] = im_t[j].squeeze()
                x_b[j, p, ...] = im_t_b[j].squeeze()

                ### gt sk and boundingbox
                y[j, p, ...] = im_s[j].squeeze()
                y_b[j, p, ...] = im_s_b[j].squeeze()

                gt_mask[j, p] = is_full[j] * 1

        num_parts_flag = torch.mean(gt_mask, 2)

        p_enc_codes = []
        p_inputs = []
        for p in range(opt.num_parts):
            input_enc = x[p, :, :, :].cuda().unsqueeze(1)
            code, p_input = ae_model[p](input_enc)  # [1,512]
            p_enc_codes.append(code)
            p_inputs.append(p_input)

        p_enc_codes = torch.cat(p_enc_codes, axis=-1)

        p_input_ = torch.cat(p_inputs, axis=1)

        # gt_masks=gt_mask.cuda()
        p_enc_codes, p_input_ = p_enc_codes.cuda(), p_input_.cuda()
        x_b = x_b.cuda()

        y = y.cuda()
        y_b = y_b.cuda()
        id_theta_parts_batch = id_theta_parts_batch.float().cuda()

        optimizer.zero_grad()
        y_hat, y_b_hat, theta_ = model(p_enc_codes, p_input_, x_b)

        bl1loss = 0
        biouloss = 0
        skloss = 0
        theta_loss = 0

        for p in range(opt.num_parts):
            y_b_ = y_b[:, p].unsqueeze(1)  # *num_parts_flag[:,p].unsqueeze(1)
            y_b_hat_ = y_b_hat[:, p].unsqueeze(1)  # *num_parts_flag[:,p].unsqueeze(1)
            biouloss += jaccard_loss(y_b_, y_b_hat_)
            bl1loss += pcn_loss_func(y_b_, y_b_hat_)
            skloss += pcn_loss_func(y[:, p].unsqueeze(1), y_hat[:, p].unsqueeze(1))
            theta_loss += aen_loss_func(theta_, id_theta_parts_batch)

        num_parts_batch = torch.sum(num_parts_flag)

        ##mask_l1_loss
        bl1loss_mean = bl1loss / num_parts_batch
        ##mask_iou_loss
        biouloss_mean = biouloss / num_parts_batch
        ##sk_loss
        skloss_mean = skloss / num_parts_batch
        ##affine_loss
        theta_loss_mean = theta_loss / num_parts_batch
        ##final loss
        loss_ = 100 * bl1loss_mean + 1 * skloss_mean + 1 * theta_loss_mean

        loss_.backward()
        optimizer.step()

        # _writer.add_image('input', torchvision.utils.make_grid(p_input_[0, :].unsqueeze(1)), i)
        # _writer.add_image('y_hat', torchvision.utils.make_grid(y_hat[0,:].unsqueeze(1)), i)
        # _writer.add_image('gt', torchvision.utils.make_grid(y[0,:].unsqueeze(1)), i)
        # _writer.add_image('y__b_hat', torchvision.utils.make_grid(y_b_hat[0,:].unsqueeze(1)), i)
        # _writer.add_image('box_gt', torchvision.utils.make_grid(y_b[0,:].unsqueeze(1)), i)

        iou_loss_display = 100 * biouloss_mean  # biouloss / num_parts_batch
        if i % opt.train_display_interval == 0:
            print('Train Epoch: {} iters:{} Loss: {:.6f}'.format(epoch, i, iou_loss_display.item()))
        torch.cuda.empty_cache()
        del y_hat, y, y_b_hat_, x_b, p_input_
        loss_sum.append(iou_loss_display.item())

    print('Train Epoch: {} \tAvg Loss: {:.6f}'.format(epoch, sum(loss_sum) / len(loss_sum)))
    # _writer.add_scalar('miou_sk__affine_loss', sum(loss_sum) / len(loss_sum),epoch)

    return sum(loss_sum) / len(loss_sum)


# def pc_one_epoch_single(model, ae_model,data_loader,opt,epoch,optimizer,writer):
#     """
#     feed data nad network: model, training_part_data
#     """
#     loss_avg=[]
#     # try:
#     aa=0
#     for i, data in enumerate(data_loader):
#         point_sets = data
#         batch_c = data[0][0].size()[0]
#         # x = torch.zeros((opt.num_parts, batch_c, 256, 256))
#         # y = torch.zeros((batch_c, opt.num_parts, 256, 256))
#         # y_mask = torch.zeros((batch_c, opt.num_parts, 256, 256))
#         im_s, im_s_b,im_n,im_n_b, is_full = point_sets[0]
#         num_nozero=torch.nonzero(is_full*1)
#         num_nozero_=len(num_nozero)
#         if num_nozero_==0:continue
#         x = torch.zeros((num_nozero_, 256, 256))
#         x_b=torch.zeros((num_nozero_, 256, 256))
#         y = torch.zeros((num_nozero_, 256, 256))
#         y_b=torch.zeros((num_nozero_, 256, 256))
#         # y_mask = torch.zeros((batch_c, 256, 256))
#         # for p in range(opt.num_parts):
#         k = 0
#         for j in range(batch_c):
#             if is_full[j]:
#                 x[k, ...] = im_n[j].squeeze()
#                 x_b[k, ...] = im_n_b[j].squeeze()
#                 y[k, ...] = im_s[j].squeeze()
#                 y_b[k, ...] = im_s_b[j].squeeze()
#                 k=k+1
#
#         code, p_gt = ae_model.encoder(x.cuda().unsqueeze(1))
#
#         # p_enc_codes = torch.cat(p_enc_codes, axis=-1)
#         # p_gts = torch.cat(p_gts, axis=1)
#
#         p_enc_codes=code.to(device).requires_grad_(True)
#         p_gts= p_gt.to(device).requires_grad_(True)
#         x_b=x_b.unsqueeze(1).to(device).requires_grad_(True)
#
#         y = Variable(y.cuda().unsqueeze(1))
#         y_b = Variable(y_b.cuda().unsqueeze(1))
#         optimizer.zero_grad()
#         y_hat,y_b_hat,trans_theta = model(p_enc_codes, p_gts,x_b)
#
#         iouloss=jaccard_loss(y_b,y_b_hat)
#
#         imloss=0.001*pcn_loss_func(y, y_hat)
#         #
#         # bbxloss= 1*pcn_loss_func(y_b, y_b_hat)
#
#         # iou_loss= iou_pytorch(y_b_hat,y_b)
#         # print("imloss:{}, bbxloss:{}".format(imloss,bbxloss))
#         # regu1=0.1*max(0,torch.sum(scaling_lowerboudn-trans_theta[:,0])) + max(0, torch.sum(trans_theta[:,0] - scaling_upperbound))
#         #
#         # regu2 =0.1*max(0, torch.sum(trans_theta[:,1] - trans_upperbound)) +max(0, torch.sum(trans_lowerbound-trans_theta[:,1]))
#         #
#         #
#         # regu3 = 0.1*max(0, torch.sum(trans_theta[:,2] - trans_upperbound))+ max(0, torch.sum(trans_lowerbound - trans_theta[:,2]))
#         loss= iouloss+imloss#bbxloss+imloss#+0.001*bbxloss
#         # loss = 10*(0.5*imloss +0.5*bbxloss)#(imloss +regu1+ regu2+ regu3) #+ 0.01* torch.norm(trans_theta,p=2))
#         loss.backward()
#
#
#         # plt.show()
#         loss_avg.append(loss.item())
#         optimizer.step()
#
#         writer.add_image('y_hat', torchvision.utils.make_grid(y_hat[0:4]), aa)
#         writer.add_image('gt', torchvision.utils.make_grid(y[0:4]), aa)
#         writer.add_image('y__b_hat', torchvision.utils.make_grid(y_b_hat[0:4]), aa)
#         writer.add_image('box_gt', torchvision.utils.make_grid(y_b[0:4]), aa)
#         aa=aa+1
#         if i % opt.train_display_interval == 0:
#             print('Train Epoch: {} iters:{} Loss: {:.6f}'.format(epoch, i, loss.item()))
#     print('Train Epoch: {} \tAvg Loss: {:.6f}'.format(epoch, sum(loss_avg) / len(loss_avg)))
#     return sum(loss_avg) / len(loss_avg)

def save_ae_model(model, pt_dir):
    print('========================saving model===============================')
    check_point = {
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
    }
    torch.save(check_point, pt_dir)
    print('========================ae model saved================================')


def save_ae_model_joint(pt_dir, opt):
    print('========================saving model===============================')
    check_points = []
    for partId in range(opt.num_parts):
        aen_pt_dir = opt.cate + '_aen_{}.pt'.format(partId)
        check_point_part = torch.load(aen_pt_dir)

        check_point = {
            'encoder': check_point_part['encoder'],
            'decoder': check_point_part['decoder'],
        }
        check_points.append(check_point)
        os.remove(aen_pt_dir)
    torch.save(check_points, pt_dir)
    print('========================ae model saved================================')


def save_pc_model(model, pt_dir):
    print('========================saving model===============================')
    check_point = {
        'trans': model.trans.state_dict(),
    }

    torch.save(check_point, pt_dir)
    print('========================pc model saved================================')


def save_pc_ae_model(ae_model, pc_model, aen_pt_dir, pcn_pt_dir, opt):
    print('========================saving model===============================')
    check_point_pcn = {
        'trans': pc_model.trans.state_dict(),
    }

    torch.save(check_point_pcn, pcn_pt_dir)
    print('========================pc model saved================================')

    check_points_aen = []
    for partId in range(opt.num_parts):
        check_point = {
            'net': ae_model[partId].state_dict(),
        }
        check_points_aen.append(check_point)

    torch.save(check_points_aen, aen_pt_dir)
    print('========================ae model saved================================')