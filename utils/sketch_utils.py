import numpy as np
from matplotlib import pyplot as plt
import cv2
from numpy.linalg import solve
import torch

def part_center(_part):
    ind_mask = np.nonzero(_part.any(axis=0))[0]
    width = ind_mask[-1] - ind_mask[0]
    centerx = int((ind_mask[-1] + ind_mask[0]) / 2)
    ind_mask = np.nonzero(_part.any(axis=1))[0]
    height = ind_mask[-1] - ind_mask[0]
    centery = int((ind_mask[-1] + ind_mask[0]) / 2)
    large_edge = max(height, width)
    bbx={'x':centerx,'y':centery,'bbx':large_edge}
    return bbx

def show_sketch(_sk,_num,_opt):
    plt.figure(1)
    for partId in range(_opt.num_parts):
        for i in range(_num):
            plt.subplot(_opt.num_parts,_num,_num*partId+i+1)
            plt.axis('off')
            if not _sk[i]['bbx'][partId] is None:
                plt.title('scale:{} cx:{} cy:{}'.format(_sk[i]['bbx'][partId]['bbx'],_sk[i]['bbx'][partId]['x'],_sk[i]['bbx'][partId]['y']))
            plt.imshow(_sk[i]['im'][partId])

            ##save part sketches
            # if i==0:
            #     cv2.imwrite('temp/' + _opt.cate + str(_opt.save_name) + '_part_'+str(partId)+'.png', (1 - _sk[i]['im'][partId]) * 255)
            # else:
            #     im=cv2.cvtColor(_sk[i]['im'][partId], cv2.COLOR_BGR2RGB)
            #     cv2.imwrite('temp/' + _opt.cate + str(_opt.save_name) + '_part_strokes'+str(partId)+'.png', im)

    plt.tight_layout()

def mask_padding(_bias,_data,_bbx,_centerx,_centery):

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


def patch_pasting(_temp,_patch,_px,_py,_pbbx):

    _raw_start, _raw_end, _col_start, _col_end = int(_py - _pbbx / 2), int(_py - _pbbx / 2) + _pbbx, int(_px - _pbbx / 2), int(_px - _pbbx / 2) + _pbbx

    if _raw_start<0:
        _temp[0:_raw_end, _col_start:_col_end] = _patch[(0-_raw_start):, :]

    elif _raw_end>256:
        _p_raw_end=_pbbx-(_raw_end-256)
        _temp[_raw_start:256, _col_start:_col_end]= _patch[0:_p_raw_end,:]

    elif _col_start <0:
        _temp[_raw_start:_raw_end, 0:_col_end] = _patch[:, (0-_col_start):]

    elif _col_end>256:
        _p_col_end = _pbbx - (_col_end - 256)
        _temp[_raw_start:_raw_end, _col_start:_p_col_end] = _patch[:,0:_p_col_end]
        pass
    else:
        _temp[_raw_start:_raw_end, _col_start:_col_end] = _patch

    return _temp


def stroke2img(sketch,imgSizes):
    canvas = np.ones((imgSizes, imgSizes, 3), dtype='uint8') * 255
    for stroke in sketch:
        for i in range(1, len(stroke)):
            penColor = (0,0,0)
            cv2.line(canvas,
                    (int(stroke[i - 1][0]), int(stroke[i - 1][1])),
                    (int(stroke[i][0]), int(stroke[i][1] )),
                    penColor)
    return np.array(canvas)

def stroke2pixel(sketch,imgSizes):
    canvas = np.ones((imgSizes, imgSizes, 3), dtype='uint8') * 255
    for stroke in sketch:
        for i in range(0, len(stroke)):
            penColor = (0,0,0)
            cv2.circle(canvas,
                      (int(stroke[i][0]), int(stroke[i][1] )),
                      1, penColor)
    return np.array(canvas)

def resize_vector(vectors,curr_center,curr_center_bbx,target_im,_normalized_scale):
    vector_=[]
    ind_mask_e = np.nonzero(target_im.any(axis=0))[0]
    centerx_e = int((ind_mask_e[-1] + ind_mask_e[0]) / 2)
    ind_mask_e = np.nonzero(target_im.any(axis=1))[0]
    centery_e = int((ind_mask_e[-1] + ind_mask_e[0]) / 2)

    for stroke_e in vectors:
        stroke_1= [(ele-np.array(curr_center)+np.array([centerx_e,centery_e]))*_normalized_scale/curr_center_bbx  for ele in stroke_e]
        vector_.append(stroke_1)
    return vector_

def get_inter(query_feature, feature_list,nearnN=3, sex=1, w_c=1, random_=-1):

    generated_f = query_feature
    feature_list =feature_list

    ## todo: recompute
    # dist_mtx = pdist_torch(generated_f, feature_list).cpu()
    # indices = np.argsort(dist_mtx, axis=1)
    # idx_sort = indices[0]


    feature_list = feature_list.cpu().numpy()
    generated_f=generated_f.cpu().numpy()

    if nearnN == 1:
        vec_mu = feature_list[0]
        vec_mu = vec_mu * w_c + (1 - w_c) * generated_f

        return  vec_mu


    # |  vg - sum( wi*vi )|   et. sum(wi) = 1
    # == | vg - v0 - sum( wi*vi) |   et. w = [1,w1,...,wn]
    A_0 = [feature_list[0, :]]
    A_m = A_0
    for i in range(1, nearnN):
        A_m = np.concatenate((A_m, [feature_list[i, :]]), axis=0)


    A_0 = np.array(A_0)
    A_m = np.array(A_m).T
    A_m0 = np.concatenate((A_m[:, 1:] - A_0.T, np.ones((1, nearnN - 1)) * 10), axis=0)

    A = np.dot(A_m0.T, A_m0)
    b = np.zeros((1, generated_f.shape[1] + 1))
    b[0, 0:generated_f.shape[1]] = generated_f - A_0

    B = np.dot(A_m0.T, b.T)

    x = solve(A, B)

    xx = np.zeros((nearnN, 1))
    xx[0, 0] = 1 - x.sum()
    xx[1:, 0] = x[:, 0]
    # print(time.time()- start_time)

    vec_mu = np.dot(A_m, xx).T * w_c + (1 - w_c) * generated_f
    vec_mu = np.array(vec_mu.astype('float32'))

    return vec_mu


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


def post_processing(_parts,_cate):
    if _cate=='chair':
        # ['chair_arm', 'chair_back', 'chair_leg', 'chair_seat']
        ##adjusting height
        leg_part_points=np.nonzero(_parts[2].any(axis=1))[0]
        seat_part_points=np.nonzero(_parts[3].any(axis=1))[0]
        back_part_points=np.nonzero(_parts[1].any(axis=1))[0]
        if len(leg_part_points)>0 and len(seat_part_points)>0:
            ##seat bottom
            seat_part_bottom = seat_part_points[-1]
            ##moving leg
            leg_part_top = leg_part_points[0]
            leg_part_bottom = leg_part_points[-1]
            dis = seat_part_bottom -leg_part_top-3
            if dis >0:
                pass
            else:
                new_part= np.zeros((256, 256), dtype="uint8")
                new_part[(leg_part_top+dis):(leg_part_bottom+dis),:]=_parts[2][leg_part_top:leg_part_bottom,:]
                _parts[2]=new_part

        if len(back_part_points)>0 and len(seat_part_points)>0:
            ##seat top
            seat_part_top = seat_part_points[0]
            ## moving back
            back_part_top =back_part_points[0]
            back_part_bottom=back_part_points[-1]

            dis = seat_part_top-back_part_bottom+2
            if dis<0:
                pass
            else:
                new_part = np.zeros((256, 256), dtype="uint8")
                new_part[(back_part_top +dis):(back_part_bottom + dis), :] = _parts[1][back_part_top:back_part_bottom, :]
                _parts[1] = new_part
    # if _cate == 'table':
    #     ##adjusting height
    #     ##['top',  'leg','stretcher','']
    #     top_part_points = np.nonzero(_parts[0].any(axis=1))[0]
    #     leg_part_points = np.nonzero(_parts[1].any(axis=1))[0]
    #     # strethcer_part_points = np.nonzero(_parts[2].any(axis=1))[0]
    #     if len(leg_part_points) > 0 and len(top_part_points) > 0:
    #         ##leg_top
    #         leg_part_top = leg_part_points[0]
    #         ## moving top
    #         top_part_top = top_part_points[0]
    #         top_part_bottom = top_part_points[-1]
    #         dis = leg_part_top - top_part_bottom + 2
    #         if dis < 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(top_part_top + dis):(top_part_bottom + dis), :] = _parts[1][top_part_top:top_part_bottom,:]
    #             _parts[0] = new_part

    # if _cate=='mug':
    #     ##adjusting width
    #     # ['handle', 'body', '', '']
    #     handle_part_points = np.nonzero(_parts[0].any(axis=0))[0]
    #     body_part_points = np.nonzero(_parts[1].any(axis=0))[0]
    #
    #     if len(handle_part_points) > 0 and len(body_part_points) > 0:
    #         ##handle_left
    #         handle_part_left=handle_part_points[0]
    #         ## moving body
    #         body_part_left = body_part_points[0]
    #         body_part_right = body_part_points[-1]
    #         dis = handle_part_left - body_part_right + 6
    #         if dis < 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(body_part_left + dis):(body_part_right + dis), :] = _parts[1][body_part_left:body_part_right, :]
    #             _parts[1] = new_part

    # if _cate == 'monitor':
    #     ##adjusting height
    #     #['base', 'stand', 'screen', ''])
    #     base_part_points = np.nonzero(_parts[0].any(axis=1))[0]
    #     stand_part_points = np.nonzero(_parts[1].any(axis=1))[0]
    #     screen_part_points = np.nonzero(_parts[2].any(axis=1))[0]
    #     if len(screen_part_points) > 0 and len(stand_part_points) > 0:
    #         ##stand top
    #         stand_part_top = stand_part_points[0]
    #         ##moving screen
    #         screen_part_top = screen_part_points[0]
    #         screen_part_bottom = screen_part_points[-1]
    #
    #         dis = stand_part_top - screen_part_bottom + 2
    #         if dis < 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(screen_part_top + dis):(screen_part_bottom + dis), :] = _parts[1][screen_part_top:screen_part_bottom, :]
    #             _parts[2] = new_part
    #
    #     if len(base_part_points) > 0 and len(stand_part_points) > 0:
    #         ##stand bottom
    #         stand_part_bottom = stand_part_points[-1]
    #         ##moving base
    #         base_part_top = base_part_points[0]
    #         base_part_bottom = base_part_points[-1]
    #         dis = stand_part_bottom - base_part_top - 3
    #         if dis > 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(base_part_top + dis):(base_part_bottom + dis), :] = _parts[2][base_part_top:base_part_bottom, :]
    #             _parts[0] = new_part

    # if _cate == 'lampa':
    #     ##adjusting height
    #     #'base', 'tube', 'shade'
    #     base_part_points = np.nonzero(_parts[0].any(axis=1))[0]
    #     tube_part_points = np.nonzero(_parts[1].any(axis=1))[0]
    #     shade_part_points = np.nonzero(_parts[2].any(axis=1))[0]
    #     if len(shade_part_points) > 0 and len(tube_part_points) > 0:
    #         ##tube top
    #         tube_part_top = tube_part_points[0]
    #         ##moving shade
    #         shade_part_top = shade_part_points[0]
    #         shade_part_bottom = shade_part_points[-1]
    #
    #         dis = tube_part_top - shade_part_bottom + 2
    #         if dis < 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(shade_part_top + dis):(shade_part_bottom + dis), :] = _parts[1][shade_part_top:shade_part_bottom, :]
    #             _parts[2] = new_part
    #
    #     if len(base_part_points) > 0 and len(tube_part_points) > 0:
    #         ##tube bottom
    #         tube_part_bottom = tube_part_points[-1]
    #         ##moving base
    #         base_part_top = base_part_points[0]
    #         base_part_bottom = base_part_points[-1]
    #         dis = tube_part_bottom - base_part_top - 3
    #         if dis > 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(base_part_top + dis):(base_part_bottom + dis), :] = _parts[2][
    #                                                                           base_part_top:base_part_bottom, :]
    #             _parts[0] = new_part

    # if _cate == 'lampc':
    #     ##adjusting height
    #     #'base', 'shade','tube'
    #     base_part_points = np.nonzero(_parts[0].any(axis=1))[0]
    #     shade_part_points = np.nonzero(_parts[1].any(axis=1))[0]
    #     tube_part_points = np.nonzero(_parts[2].any(axis=1))[0]
    #
    #     if len(shade_part_points) > 0 and len(tube_part_points) > 0:
    #         ##tube bottom
    #         tube_part_bottom = tube_part_points[-1]
    #         ##moving shade
    #         shade_part_top = shade_part_points[0]
    #         shade_part_bottom = shade_part_points[-1]
    #
    #         dis = tube_part_bottom - shade_part_top - 3
    #         if dis > 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(shade_part_top + dis):(shade_part_bottom + dis), :] = _parts[1][
    #                                                                             shade_part_top:shade_part_bottom, :]
    #             _parts[1] = new_part
    #
    #     if len(base_part_points) > 0 and len(tube_part_points) > 0:
    #         ##tube bottom
    #         tube_part_bottom = tube_part_points[-1]
    #         ##moving base
    #         base_part_top = base_part_points[0]
    #         base_part_bottom = base_part_points[-1]
    #         dis = tube_part_bottom - base_part_top - 3
    #         if dis > 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(base_part_top + dis):(base_part_bottom + dis), :] = _parts[2][
    #                                                                           base_part_top:base_part_bottom, :]
    #             _parts[0] = new_part

    # if _cate == 'car':
    #     ##adjusting height
    #     # 'body', 'wheel', 'mirror'
    #     body_part_points = np.nonzero(_parts[0].any(axis=1))[0]
    #     wheel_part_points = np.nonzero(_parts[1].any(axis=1))[0]
    #     mirror_part_points = np.nonzero(_parts[2].any(axis=1))[0]
    #     if len(wheel_part_points) > 0 and len(body_part_points) > 0:
    #         ##body bottom
    #         body_part_bottom = body_part_points[-1]
    #         ##moving wheel
    #         wheel_part_top = wheel_part_points[0]
    #         wheel_part_bottom = wheel_part_points[-1]
    #         wheel_middel=int((wheel_part_top+wheel_part_bottom)/2)
    #         dis = body_part_bottom - wheel_middel - 3
    #         if dis > 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(wheel_part_top + dis):(wheel_part_bottom + dis), :] = _parts[2][wheel_part_top:wheel_part_bottom, :]
    #             _parts[1] = new_part

    # if _cate == 'guitar':
    #     ##adjusting height
    #     # 'head', 'neck', 'body'
    #     head_part_points = np.nonzero(_parts[0].any(axis=1))[0]
    #     neck_part_points = np.nonzero(_parts[1].any(axis=1))[0]
    #     body_part_points = np.nonzero(_parts[2].any(axis=1))[0]
    #
    #     if len(head_part_points) > 0 and len(neck_part_points) > 0:
    #         ##neck top
    #         neck_part_top = neck_part_points[0]
    #         ##moving head
    #         head_part_top = head_part_points[0]
    #         head_part_bottom = head_part_points[-1]
    #
    #         dis = neck_part_top - head_part_bottom + 2
    #         if dis < 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(head_part_top + dis):(head_part_bottom + dis), :] = _parts[1][head_part_top:head_part_bottom, :]
    #             _parts[0] = new_part
    #
    #
    #
    #     if len(body_part_points) > 0 and len(neck_part_points) > 0:
    #         ##neck bottom
    #         neck_part_bottom = neck_part_points[-1]
    #         ##moving body
    #         body_part_top = body_part_points[0]
    #         body_part_bottom = body_part_points[-1]
    #
    #         dis = neck_part_bottom - body_part_top - 10
    #         if dis > 0:
    #             pass
    #         else:
    #             new_part = np.zeros((256, 256), dtype="uint8")
    #             new_part[(body_part_top + dis):(body_part_bottom + dis), :] = _parts[2][body_part_top:body_part_bottom, :]
    #             _parts[2] = new_part

    if _cate=='vase':
        # 'top', 'handle', 'body', 'base'
        top_part_points = np.nonzero(_parts[0].any(axis=1))[0]
        handle_part_points = np.nonzero(_parts[1].any(axis=1))[0]
        body_part_points = np.nonzero(_parts[2].any(axis=1))[0]
        base_part_points = np.nonzero(_parts[3].any(axis=1))[0]
        if len(top_part_points) > 0 and len(body_part_points) > 0:
            ##seat top
            body_part_top = body_part_points[0]
            ## moving back
            top_part_top = top_part_points[0]
            top_part_bottom = top_part_points[-1]

            dis = body_part_top - top_part_bottom + 4
            if dis < 0:
                pass
            else:
                new_part = np.zeros((256, 256), dtype="uint8")
                new_part[(top_part_top + dis):(top_part_bottom + dis), :] = _parts[0][top_part_top:top_part_bottom,:]
                _parts[0] = new_part



    return _parts





def vector_transfrom():
    print()
    pass