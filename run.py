import argparse
import ast
from dca.trainer import Train
from dca.evaluate import Eval
from dca.utils import setup_runtime

# runtime arguments
### training or testing ###
parser = argparse.ArgumentParser(description='Configurations.')
parser.add_argument('--config', default='./experiments/train_vari.yml', type=str, help='Specify a config file path')
### device config ###
parser.add_argument('--seed', default=None, type=int, help='Specify a random seed')
parser.add_argument('--num_workers', default=8, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('--device', default='gpu', type=str, help='Specify running device')
parser.add_argument('--multi_gpu', default=True, type=ast.literal_eval, help='Use multi gpus or not, only works when using gpu and during training')
parser.add_argument('--device_id', default=[0, 1], type=list, help='Specify gpus')
args = vars(parser.parse_args())  # convert args to dict

# set up
cfgs = setup_runtime(args)
run_train = cfgs.get('run_train', False)

# run
if run_train:
    trainer = Train(cfgs)
    trainer.train()
else:
    eval = Eval(cfgs)
    # eval.test_real_images()
    # eval.test_sun()
    eval.test()
    # eval.vis()
