import time
import os
import json

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
                        p_list_arc = resample(np.array(ele_list), inc=10)
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
        img_sequence.append(sk_data)
    return img_sequence