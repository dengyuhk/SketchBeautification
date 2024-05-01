from pathlib import Path
import copy
# import matplotlib
# import tkinter
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from p_assembly_module.models import parts_aes_graphs,get_model_pcn
from utils.sample_arclength import resample
from utils.sketch_utils import part_center,show_sketch,mask_padding,resize_vector,stroke2pixel,stroke2img,get_inter,vector_transfrom,patch_pasting
import argparse
import os
import cv2
import json
import torch
import pickle
from PIL import Image
from skimage.morphology import skeletonize
from numpy.linalg import solve
from utils.dynamic_2d_registration import dynamic_2d_registration_deng
from utils.vector_resize import sample_point2,get_final_stroke,trans,drawColor
from utils.scipy_optimization import *
from scipy.spatial.distance import cdist
# from utils.sketch_utils import post_processing

parser = argparse.ArgumentParser(description='code_generation')
parser.add_argument('--cate', type=str, default='chair')
parser.add_argument('--normalized_scale', type=int, default=128)
parser.add_argument('--code_folder', type=str, default='code_books')
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--root', type=str, default='')
parser.add_argument('--num_parts', type=int, default=4)
parser.add_argument('--.sk_representation', type=str, default='')
opt = parser.parse_args()


use_cuda = torch.cuda.is_available()



def array2samples_distance(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
    """
    num_point2, num_features = array2.shape
    expanded_array1 = np.tile(array1, (num_point2, 1))
    num_point1, num_features = array1.shape
    expanded_array2 = np.reshape(np.tile(np.expand_dims(array2, 1),(1, num_point1, 1)),(-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point1, num_point2))
    try:
        distances = np.min(distances, axis=1)
    except:
        print(expanded_array1)
        print(expanded_array2)
        print(distances)
        raise('error')


    # distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    # batch_size, num_point, num_features = array1.shape
    dist = 0
    # for i in range(batch_size):
    av_dist1 = array2samples_distance(array1, array2)
    av_dist2 = array2samples_distance(array2, array1)
    dist = dist + (av_dist1+av_dist2)
    return dist

def IDW(_dist_mtx_topK,_listTopK):
    sum0=0
    sum1=0
    for _Sindex in range(len(_listTopK)):
        sum0+=_listTopK[_Sindex]/_dist_mtx_topK[_Sindex]
        sum1+=1/_dist_mtx_topK[_Sindex]

    return sum0/sum1

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

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx

def search_Inte_torch(_query_im, _query_code, _code_book, _im_boook, opt):

    _code_book = torch.from_numpy(_code_book).cuda()
    dist_mtx = pdist_torch(_query_code, _code_book).cpu()
    indices = np.argsort(dist_mtx, axis=1)
    ##todo: test for the novel image


    ## here we rerank the distance according to the chamfer distance
    cd_samples=40
    cd_topK = []

    p_a = np.asarray(np.nonzero(_query_im)).T
    latent_topK_rerank_ = _code_book[indices[0][0:cd_samples]]

    for Qindex in range(opt.startId,  cd_samples):  # (len(indices[0])):
        sindex = indices[0][Qindex]
        temp = _im_boook[sindex]
        chamfer_dist_e = chamfer_distance_numpy(p_a,temp)
        cd_topK.append(chamfer_dist_e)

    cd_topK=np.array(cd_topK)
    cd_order=np.argsort(cd_topK)

    rerank_latent_topK=latent_topK_rerank_[cd_order[0:opt.topK]]


    try:
        latent_topK_interp_lli_rerank = get_inter(_query_code, rerank_latent_topK, nearnN=opt.topK)
    except:
        dist_mtx_topK=cd_topK[cd_order[0:opt.topK]]
        latent_topK_interp_lli_rerank = IDW(dist_mtx_topK, rerank_latent_topK).cpu().numpy()

    return latent_topK_interp_lli_rerank

def closest_node(node,nodes):
    return [nodes[cdist(node[node_e][np.newaxis,:],nodes).argmin()] for node_e in range(len(node))]

def search_and_interp(pinput,opt):
    ## model:
    #sk_data = {'im': sk_im, 'vec': sk_vector,'name':file_list[skname],'bbx':sk_bbxs,'pFlags':sk_pFlags}

    sk_p_beauty=copy.deepcopy(pinput).copy()
    for partId in range(opt.num_parts):
        if sk_p_beauty['pFlags'][partId]==False:
            continue

        code_book=opt.books[partId]['latent']
        # im_book = opt.books[partId]['gt']
        cham_pt_book=opt.books[partId]['cham_pt']
        test_im = sk_p_beauty['im'][partId]
        test_vector = sk_p_beauty['vec'][partId]
        part_data_gpu = torch.from_numpy(test_im).cuda().unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            z_vector, _ = opt.p_search_model[partId](part_data_gpu, None, None, is_training=False)
            ## retireval and interpolation
            local_linear_interp_rerank = search_Inte_torch(test_im, z_vector.double(), code_book, cham_pt_book, opt)
            reteival_interp_l_rerank = torch.tensor(local_linear_interp_rerank).float().squeeze().cuda().unsqueeze(0)

            ## reconstruction
            frame_flag_loc_rerank = np.zeros([opt.normalized_scale, opt.normalized_scale], np.uint8)
            # get frame grid values
            point_coord = opt.coords[np.newaxis, :, :].astype(np.float32)
            point_coord = torch.from_numpy(point_coord).cuda()

            #feed point coordinates to obtain the correspoidng point sets
            _, model_out_ = opt.p_search_model[partId](None, reteival_interp_l_rerank, point_coord, is_training=False)
            model_out = model_out_.detach().cpu().numpy()  # [0]
            frame_flag_loc_rerank[opt.x_coords, opt.y_coords] = np.reshape((model_out > opt.sampling_threshold).astype(np.uint8),[opt.test_point_batch_size])  # self.sampling_threshold

            p_a = np.asarray(np.nonzero(frame_flag_loc_rerank)).T
            p_a_ = p_a.copy()
            p_a_[:, 0] = p_a[:, 1]
            p_a_[:, 1] = p_a[:, 0]
            ## todo: first registration
            stroke_query = []
            for stroke_e in test_vector:
                stroke_query = stroke_query + stroke_e
            stroke_query_np = np.array(stroke_query)
            stroke_query_np_ = stroke_query_np.copy()
            stroke_query_np_[:, 0] = stroke_query_np[:, 1]
            stroke_query_np_[:, 1] = stroke_query_np[:, 0]

            stroke_registered, idx = dynamic_2d_registration_deng(stroke_query_np_, p_a)

            ## todo: curve optimization distance
            corres_top = []
            corres_top_op = []
            stroke_registered_ = stroke_registered.copy()
            stroke_registered_[:, 0] = stroke_registered[:, 1]
            stroke_registered_[:, 1] = stroke_registered[:, 0]
            stroke_len = 0
            for stroke_e in test_vector:
                stroke_len_e = len(stroke_e)

                corres_stroke_e = p_a_[idx[stroke_len:stroke_len_e + stroke_len], :].tolist()
                corres_top.append(corres_stroke_e) ## corresponding curves

                corres_stroke_np, source_stroke_np = np.array(corres_stroke_e), np.array(stroke_e)
                stroke_op_e = optim_dy(source_stroke_np, corres_stroke_np).tolist()
                corres_top_op.append(stroke_op_e)  ## optimizaed curves
                stroke_len = stroke_len_e + stroke_len

        sk_p_beauty['vec'][partId]=corres_top_op
        points, labels, order = sample_point2(corres_top_op, range(len(corres_top_op)))
        final_stroke, final_label = get_final_stroke(points, labels, order)
        final = trans(final_stroke, final_label)
        final_im = drawColor(final, opt.normalized_scale + 15)
        sk_p_beauty['im'][partId]=final_im

    return sk_p_beauty

def initialize_Assembly(_opt):
    # assembly network
    aen = parts_aes_graphs(_opt.num_parts)

    if use_cuda:
        for i in range(opt.num_parts):
            aen[i] = aen[i].encoder.cuda()

    ## reload the weights of the aen
    ass_ae_pt_dir = os.path.join(BASE_DIR, 'trained_models', opt.cate + '_aen_local.pt')
    check_point = torch.load(ass_ae_pt_dir)
    for partId in range(opt.num_parts):
        aen[partId].load_state_dict(check_point[partId]['net'])

    pcn = get_model_pcn(_opt.num_parts)
    if use_cuda:
        pcn = pcn.cuda()

    ## reload the weights of the aen
    ass_pc_pt_dir = os.path.join(BASE_DIR, 'trained_models', opt.cate + '_pcn_local.pt')
    check_point = torch.load( ass_pc_pt_dir)
    pcn.trans.load_state_dict(check_point['trans'])

    return aen, pcn

def post_processing(_parts,_opt):
    if _opt.cate=='chair':
        # ['chair_arm', 'chair_back', 'chair_leg', 'chair_seat']
        leg_part_points=np.nonzero(_parts[2].any(axis=1))[0]
        seat_part_points=np.nonzero(_parts[3].any(axis=1))[0]
        if len(leg_part_points)>0 and len(seat_part_points)>0:
            ##seat bottom
            seat_part_bottom = seat_part_points[-1]
            ##leg top
            leg_part_top = leg_part_points[0]
            leg_part_bottom = leg_part_points[-1]
            dis = seat_part_bottom -leg_part_top-3
            if dis >0:
                pass
            else:
                new_part= _opt.img_none.copy()
                new_part[(leg_part_top+dis):(leg_part_bottom+dis),:]=_parts[2][leg_part_top:leg_part_bottom,:]
                _parts[2]=new_part

    return _parts

def sk_assembly(_sk,_opt):


    with torch.no_grad():
        x = torch.zeros((_opt.num_parts, 1, 256, 256))
        x_b = torch.zeros((1, opt.num_parts, 256, 256))


        for p in range(_opt.num_parts):
            x[p, 0, ...] = torch.tensor(_sk['im'][p]).squeeze()

        p_enc_codes = []
        p_gts = []
        for p in range(opt.num_parts):
            input_enc = x[p, :, :, :].cuda().unsqueeze(1).detach()
            code, p_gt = opt.aen[p](input_enc)
            p_enc_codes.append(code.detach())
            p_gts.append(p_gt.detach())

        p_enc_codes = torch.cat(p_enc_codes, axis=-1)
        p_gt_ = torch.cat(p_gts, axis=1)

        p_enc_codes, p_gt_ = p_enc_codes.cuda(), p_gt_.cuda()
        x_b = x_b.cuda()

        y_hat, y_b_hat, theta_ = opt.pcn(p_enc_codes, p_gt_, x_b)

        # for n_time in range(2):
        #     ## re initalization
        #     x=y_hat.transpose(0,1)
        #     x_b= y_b_hat
        #     p_enc_codes = []
        #     p_gts = []
        #     for p in range(opt.num_parts):
        #         input_enc = x[p, :, :, :].cuda().unsqueeze(1).detach()
        #         code, p_gt = opt.aen[p](input_enc)  # 1,512
        #         p_enc_codes.append(code.detach())
        #         p_gts.append(p_gt.detach())
        #
        #     p_enc_codes = torch.cat(p_enc_codes, axis=-1)
        #     p_gt_ = torch.cat(p_gts, axis=1)
        #
        #     p_enc_codes, p_gt_ = p_enc_codes.cuda(), p_gt_.cuda()
        #     x_b = x_b.cuda()
        #
        #     y_hat, y_b_hat, theta_ = opt.pcn(p_enc_codes, p_gt_, x_b)

        strokes_mask=np.zeros((256, 256), dtype="uint8")
        strokes_ = 255*np.ones((256, 256,3), dtype="uint8")

        parts=[]
        parts_sk=[]
        for parts_id in range(_opt.num_parts):
            part = y_hat[0, parts_id].detach().cpu().squeeze().numpy()
            part[part>0]=1
            parts.append(part)
            parts_sk.append(skeletonize(part))
        parts=post_processing(parts,_opt)
        color_map=[[255, 0, 0], [0, 128, 128], [0, 255, 0], [128, 0, 128]]
        for partId in range(len(parts)):
            strokes_[parts[partId] > 0] = color_map[partId]


            strokes_mask[parts_sk[partId] > 0] = 1
            # plt.subplot(121)
            # plt.imshow(parts[partId])
            # plt.subplot(122)
            # plt.imshow(parts_sk[partId])
            # plt.show()
        strokes_mask=skeletonize(strokes_mask)
    return strokes_mask


def initialize_Search_interp(_opt):
    print('## initializing........{}...........'.format(_opt.cate))
    print('loading network.....................')

    ## network
    # implicit_network
    aes_pt_dir = os.path.join(BASE_DIR, 'trained_models', _opt.cate + '_im.pt')
    p_search_model = torch.load(aes_pt_dir)
    if use_cuda:
        for i in range(_opt.num_parts):
            p_search_model[i] = p_search_model[i].cuda()

    ## code_book:
    print('loading code_book.....................')
    _code_folder = _opt.code_folder
    predict_path_ztest = os.path.join('part_retrieval', _code_folder,'{}_{}_{}.pkl'.format(_opt.cate, _opt.net_type, 'latent_cham_pt'))#latent #latent_im

    if Path(predict_path_ztest).is_file():
        with Path(predict_path_ztest).open('rb') as fh:
            books = pickle.load(fh)

    ##hyperparameters setting
    opt.test_point_batch_size = opt.normalized_scale * opt.normalized_scale
    opt.coords = np.zeros((opt.normalized_scale * opt.normalized_scale, 2), np.uint8)
    _point_count = 0
    opt.sampling_threshold = 0.5
    for i in range(0, opt.normalized_scale):
        for j in range(0, opt.normalized_scale):
            opt.coords[_point_count, 0] = i
            opt.coords[_point_count, 1] = j
            _point_count = _point_count + 1
    opt.x_coords = opt.coords[0:opt.test_point_batch_size][:, 0]
    opt.y_coords = opt.coords[0:opt.test_point_batch_size][:, 1]

    opt.im_zero = np.zeros((opt.normalized_scale, opt.normalized_scale))
    opt.img_none=np.zeros((256, 256), dtype="uint8")

    return p_search_model, books

def readimg(_path,_opt):## add exsiting flag and add the boudning box
    img_sequence = []
    file_list = os.listdir(_path)
    num = len(file_list)
    for skname in range(0, num):
        sq_path = os.path.join(_path, file_list[skname])
        with open(sq_path) as json_file:
            data = json.load(json_file)
        sk_im = []
        sk_vector = []
        sk_bbxs=[]
        sk_pFlags=[]
        for pId in range(_opt.num_parts):
            exist_flag=[stroke_e['stroke_class']  for stroke_e in data if stroke_e['stroke_class'] == pId ]
            sk_list_arc = []
            if len(exist_flag)==0:
                pass
            else:
                for stroke_e in data:
                    ele_list = []
                    if stroke_e['stroke_class'] == pId:
                        for ele in stroke_e['path']:
                            ele_resize = [ele_e * 256 / 512 for ele_e in ele]
                            if ele_resize not in ele_list:  # remove the duplicate elements
                                ele_list.append(ele_resize)
                        p_list_arc = resample(np.array(ele_list), inc=10)#10
                        sk_list_arc.append(p_list_arc)
                    else:
                        pass

            if len(sk_list_arc) != 0:
                img = stroke2img(sk_list_arc, 256)[:, :, 0]
                im = (img != 255)
                _part_data = im * 1
                bbx=part_center(_part_data)
                bias = 0
                _data_temp_e_ = (mask_padding(bias, _part_data, bbx['bbx'] + 4, bbx['x'], bbx['y']) * 2).astype('float32')
                vec_resize = resize_vector(sk_list_arc, [bbx['x'], bbx['y']], bbx['bbx'] + 4, _data_temp_e_,opt.normalized_scale)
                img_vector = (stroke2img(vec_resize, _opt.normalized_scale)[:, :, 0] != 255) * 1
                sk_pFlags.append(True)
                sk_bbxs.append(bbx)
                sk_vector.append(vec_resize)
                sk_im.append(img_vector)
            else:
                sk_pFlags.append(False)
                sk_bbxs.append(None)
                sk_im.append(opt.im_zero)
                sk_vector.append([])
        sk_data = {'im': sk_im, 'vec': sk_vector,'name':file_list[skname],'bbx':sk_bbxs,'pFlags':sk_pFlags}
        if _opt.cate == 'airplane':
            temp = sk_data['vec'][3].copy()
            sk_data['vec'][3]=sk_data['vec'][1]
            sk_data['vec'][1]=temp

            temp1 = sk_data['im'][3].copy()
            sk_data['im'][3] = sk_data['im'][1]
            sk_data['im'][1] = temp1

            temp2 = sk_data['bbx'][3].copy()
            sk_data['bbx'][3] = sk_data['bbx'][1]
            sk_data['bbx'][1] = temp2

            temp3 = sk_data['pFlags'][3]
            sk_data['pFlags'][3] = sk_data['pFlags'][1]
            sk_data['pFlags'][1] = temp3


            # for pId in range(_opt.num_parts):
            #     print(pId)
            #     print(_opt.part_names[pId])
            #     plt.imshow( sk_data['im'][pId])
            #     plt.show()
            #     print()
        img_sequence.append(sk_data)
    return img_sequence

def sk_reformart(_sk,_opt):

    _canvas=np.zeros((256,256), dtype="uint8")
    sk_combine=copy.deepcopy(_sk)
    for partId in range(_opt.num_parts):
        if sk_combine['pFlags'][partId]==True:

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
                part_data= cv2.resize(im_content, (pbbx, pbbx), interpolation=cv2.INTER_NEAREST)*2
                part_data = skeletonize(np.asarray(part_data) > 0) * 1
            else:
                part_data = cv2.resize(im_content, (pbbx, pbbx), interpolation=cv2.INTER_AREA)*2
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
            sk_combine['im'][partId] = _opt.img_none
    sk_combine['full_im']=_canvas
    return sk_combine


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

    # BASE_DIR = '/home/wanchao/new_drive/sk_beautification'
    BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(BASE_DIR, 'data/sketch_out')
    user_drawings_path='sequence'
    opt.root = root


    ##todo: ===================================================================
    opt.net_type ='search'#'search' # 'interp'
    opt.fig_viusal_flag=True
    opt.topK=3
    opt.startId=0
    opt.im_zero = np.zeros((opt.normalized_scale, opt.normalized_scale))


    im_cates={'3.json':'chair', '53.json':'guitar', '54.json':'guitar','78.json':'airplane','141.json':'vase'}
    test_im_book = readimg(user_drawings_path, opt)
    save_name=0
    for file_id in range(len(test_im_book)):

        _test_im_book_e = test_im_book[file_id]
        opt.save_name=_test_im_book_e['name']
        print('processing {}'.format(_test_im_book_e['name']))

        opt.cate = im_cates[_test_im_book_e['name']]
        opt.num_parts = len(cate_parts[opt.cate])
        opt.part_names = cate_parts[opt.cate]
        opt.p_search_model, opt.books = initialize_Search_interp(opt)
        opt.aen, opt.pcn = initialize_Assembly(opt)

        sketch_p_beauty=search_and_interp(_test_im_book_e,opt)
        compare_list=[_test_im_book_e,sketch_p_beauty]
        show_sketch(compare_list,len(compare_list),opt)
        if not os.path.exists('output'): os.mkdir('output')
        plt.savefig('output/'  +opt.cate+ str(opt.save_name) + '_1.png', bbox_inches="tight", pad_inches=0)
        sk_combine=sk_reformart(sketch_p_beauty, opt)

        sk_beauty=sk_assembly(sk_combine,opt)

        plt.figure()
        plt.subplot(131)
        original_combine=sk_reformart(_test_im_book_e,opt)
        plt.imshow(original_combine['full_im'])
        cv2.imwrite('output/'  +opt.cate + str(opt.save_name) + '_ori.png',(1-original_combine['full_im'])*255)
        plt.subplot(132)
        plt.imshow(sk_combine['full_im'])
        cv2.imwrite('output/'  +opt.cate + str(opt.save_name) + '_sk_beauty.png', (1 - sk_combine['full_im']) * 255)
        plt.subplot(133)
        plt.imshow(sk_beauty)
        cv2.imwrite('output/'  +opt.cate + str(opt.save_name) + '_sk_assembly.png', (1-sk_beauty*1) * 255)
        plt.savefig('output/'  +opt.cate + str(opt.save_name) + '_2.png', bbox_inches="tight", pad_inches=0)
        # save_name = save_name + 1

        # sk_final=Image.fromarray(sk_beauty)
        # sk_final=sk_final.resize((512,512),Image.ANTIALIAS)
        # plt.figure(4)
        # plt.imshow(sk_final)
        plt.show()
        print()


