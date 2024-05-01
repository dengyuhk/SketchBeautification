import numpy as np
import cv2
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

def sample(points,s):
    """
    :param points: points[P0,P1,Pn-1] raw points
    :param s: arclength
    :return: point on P ar arclength s
    """
    cur_length = 0
    for i in range(len(points) - 1):
        length = np.linalg.norm(points[i + 1] - points[i])
        if cur_length + length >= s:
            alpha = (s - cur_length) / length
            result = lerp(points[i], points[i + 1], alpha)
            return result
        cur_length += length

def resample(points, inc=0.01):
    """
    Given:
        points: [P0, P1, ..., Pn-1] raw points
        inc: sampling rate of the curve, 1cm
             can modify to other sampling rate, e.g. how many points
    """
    length = 0
    for i in range(len(points) - 1):
        length += np.linalg.norm(points[i + 1] - points[i])

    num = int(length / inc)

    q = []
    for i in range(num):
        q.append(sample(points, i * inc))

    q.append(points[-1,:])


    return q



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

def stroke2pixel(sketch,imgSizes):
    canvas = np.ones((imgSizes, imgSizes, 3), dtype='uint8') * 255
    for stroke in sketch:
        for i in range(0, len(stroke)):
            penColor = (0,0,0)
            cv2.circle(canvas,
                      (int(stroke[i][0]), int(stroke[i][1] )),
                      1, penColor)
    return np.array(canvas)

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

def show_sketch(_sk,_num,_num_parts):
    plt.figure(1)
    for partId in range(_num_parts):
        for i in range(_num):
            plt.subplot(_num_parts)
            plt.axis('off')
            if not _sk[i]['bbx'][partId] is None:
                plt.title('scale:{} cx:{} cy:{}'.format(_sk[i]['bbx'][partId]['bbx'],_sk[i]['bbx'][partId]['x'],_sk[i]['bbx'][partId]['y']))
            plt.imshow(_sk[i]['im'][partId])

    plt.tight_layout()