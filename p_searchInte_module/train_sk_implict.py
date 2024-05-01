import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from framework import IM_AE
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default= 0.000005369, type=float, help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--cate", action="store", dest="cate", default="airplane", help="The name of dataset")
parser.add_argument("--part_name", action="store", dest="part_name", default="airplane_body", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="data/sketch_out/", help="Root directory of dataset [data]")
parser.add_argument('--lr_policy', type=str, default='step')
parser.add_argument('--lr-decay-iters', type=int, default=40,help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_im_size", action="store", dest="sample_im_size", default=128, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")#default is False[deng]
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--gpuId", action="store", dest="gpuId", default=0, type=int, help="0,1 gpu devices")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]") #default is False[deng]
parser.add_argument("--svr", action="store_true", dest="svr", default=False, help="True for svr [False]")
parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
parser.add_argument("--interp", action="store_true", dest="interp", default=False, help="True for getting latent codes [False]")
FLAGS = parser.parse_args()


def run(_ae_flag,_train_flag,_sample_dir_flag,_start_flag,_end_flag,_epoch_flag,_getz_flag,_interp_flag,_checkpoint_dir_flag,part_name):


	FLAGS.ae=_ae_flag
	FLAGS.train=_train_flag
	FLAGS.sample_dir=_sample_dir_flag
	FLAGS.start=_start_flag
	FLAGS.end = _end_flag
	FLAGS.epoch=_epoch_flag
	FLAGS.getz=_getz_flag
	FLAGS.interp=_interp_flag
	FLAGS.checkpoint_dir=_checkpoint_dir_flag
	FLAGS.part_name=part_name

	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	if FLAGS.ae:
		im_ae = IM_AE(FLAGS)

		if FLAGS.train:
			im_ae.train(FLAGS)
		elif FLAGS.getz:
			im_ae.get_z(FLAGS)
		elif FLAGS.interp:
			im_ae.interpolation(FLAGS)
		else:
			im_ae.test_reconstruction(FLAGS)
	else:
		print("Please specify an operation: ae or svr?")

if __name__ == '__main__':
	## hypyparameters
	part_name = 'chair_leg'#'Car_labelA'#'chair_seat'#'Guitar_labelA'#'chair_leg'  #

	## traning parameters
	_ae_flag=True
	_train_flag=False
	_interp_flag=True
	_getz_flag = False
	_sample_dir_flag='samples/im_ae_out_'+part_name
	_checkpoint_dir_flag='checkpoint'
	_start_flag=0
	_end_flag=10

	_epoch_flag=1000
	run(_ae_flag,_train_flag,_sample_dir_flag,_start_flag,_end_flag,_epoch_flag,_getz_flag,_interp_flag,_checkpoint_dir_flag,part_name)
