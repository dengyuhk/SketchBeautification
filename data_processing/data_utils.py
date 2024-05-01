import os
import numpy as np
from data_processing import part_dataset_pcn_local
from data_processing import part_dataset_searInte
import torch
from matplotlib import pyplot as plt



def load_pcn_data_local(data_path, category, phase, batch_size, disorderKey,trans_flag=False):
    print('Loading local assembly training data')
    pcn_dataset = part_dataset_pcn_local.PartDatasetPCN(root=data_path, transform_flag=trans_flag,class_choice=category, split=phase)
    num_parts = pcn_dataset.get_number_of_parts()
    data_loader = torch.utils.data.DataLoader(pcn_dataset,
            batch_size=batch_size,
            shuffle=True if phase[:5]=='train' and  disorderKey else False,
             num_workers=4)

    return data_loader, num_parts



def load_psi_data(data_path, category, phase, num_parts,part_labels,normalized_scale,batch_size,disorderKey,_sk_representation=''):
    """
    Utilize a for loop to read the different part data
    input: datapath, [train val]_index, the total_part_num
    output: the list of part_data(at least 4 elements)
    """
    # data_loader_p=[]

    # for i in range(num_parts):
    #     print('Loading part ' + str(i))
    #
    #     psi_datset=part_dataset_searInte.PartDatasetSearInte(_dataPath=data_path, _cate=category,_normalized_scale=normalized_scale, _phase=phase,_part_label=i,_sk_representation=_sk_representation)
    #     data_loader_p.append(torch.utils.data.DataLoader(psi_datset,batch_size=batch_size,shuffle=True if phase[:5]=='train' and  disorderKey else False,
    #          num_workers=4)
    #     )
    # for i in range(num_parts):
    print('Loading part ' + str(part_labels))

    psi_datset = part_dataset_searInte.PartDatasetSearInte(_dataPath=data_path, _cate=category, _normalized_scale=normalized_scale, _phase=phase,_part_label=part_labels, _sk_representation=_sk_representation)
    # data_loader_p.append(torch.utils.data.DataLoader(psi_datset, batch_size=batch_size, shuffle=True if phase[:5] == 'train' and disorderKey else False,num_workers=4))

    return psi_datset#data_loader_p#, ae_test_dataset

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cate='airplane'#'chair'
    seen_split='train'

    ## todo: test the pcn_loader

    train_pcn_loader,num_parts=load_pcn_data_local(os.path.join(BASE_DIR, 'data/sketch_out'), cate, seen_split,batch_size=16,disorderKey=True,trans_flag=True)

    for i, batch in enumerate(train_pcn_loader):

        gts=batch[0]
        trans=batch[1]
        flags=batch[2]
        print("The {}th batch".format(i))
        plt.figure(1, figsize=(10, 8))
        m, n = len(gts), 4
        sample_id=0
        plt.figure(1)
        for parts_id in range(len(gts)):
            plt.subplot(m, n, n * parts_id + 1)
            plt.axis('off')
            plt.title('gt')
            plt.imshow(gts[parts_id][sample_id][:,:,0])

            plt.subplot(m, n, n * parts_id + 2)
            plt.axis('off')
            plt.title('rand_affine')
            plt.imshow(trans[parts_id][sample_id][:,:,0])

            plt.subplot(m, n, n * parts_id + 3)
            plt.axis('off')
            plt.title('gt')
            plt.imshow(gts[parts_id][sample_id][:, :, 1])
            #
            plt.subplot(m, n, n * parts_id + 4)
            plt.axis('off')
            plt.title('rand_affine')
            plt.imshow(trans[parts_id][sample_id][:, :, 1])
            # plt.subplot(m, n, n * parts_id + 3)
            # plt.axis('off')
            # plt.imshow(batch[parts_id][0][sample_id + 2])
            # plt.subplot(m, n, n * parts_id + 4)
            # plt.axis('off')
            # plt.imshow(batch[parts_id][0][sample_id + 3])
            # note[deng]: the first dimension 4 list for 4 parts, we have 4 * batchsize;
            # the second dimension, means [gt, input]
            # the third dimension indicates the different samples.
        plt.tight_layout()

        plt.figure(2)
        strokes_ = np.zeros((256, 256), dtype="uint8")
        for parts_id in range(m):
            part =gts[parts_id][sample_id][:,:,0].numpy()
            strokes_[part > 0] = 1
        plt.subplot(121)
        plt.axis('off')
        plt.title('ground_truth')
        plt.tight_layout()
        plt.imshow(255*(1-strokes_),cmap='gray')

        strokes_ = np.zeros((256, 256), dtype="uint8")
        for parts_id in range(m):
            part = trans[parts_id][sample_id][:, :, 0].numpy()
            strokes_[part > 0] = 1
        plt.subplot(122)
        plt.axis('off')
        plt.title('random_affine')
        plt.tight_layout()
        plt.imshow(255*(1-strokes_),cmap='gray')
        plt.show()
        

    ##todo: test the aen_loader
    # train_aen_loaders = load_aen_data(os.path.join(BASE_DIR, 'data/sketch_out'), cate, seen_split, num_parts=4,
    #                                  batch_size=16, disorderKey=True)
    # for train_aen_loader in train_aen_loaders:
    #     for i, batch in enumerate(train_aen_loader):
    #         if i >2:continue
    #         print("The {}th batch".format(i))
    #         ## display data
    #         sample_id = 0
    #         plt.imshow(batch[sample_id].squeeze(), cmap='gray')
    #         plt.show()


