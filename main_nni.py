import os
import torch
import nni
import argparse
from trainers.trainer_origin_target import Trainer
# from trainers.trainer import Tester
from data_loader.data_generator import DataGenerator
from data_loader.data_generator import shuffle
from utils.utils import get_args
from utils.config import get_config_from_json
# from utils.logger import Logger
from models.model_xavier_init import Net


class Config():
    def __init__(self):
        pass

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="31_train")
    parser.add_argument("--model_name", type=str, default="m16 test batchsize")
    parser.add_argument("--data_dir",type=str, default="../LQ_SRP_SmartMeter/data_split")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--scale_factor", type=int, default=10)
    parser.add_argument("--num_epochs",type=int, default=100)
    parser.add_argument("--save_epochs",type=int, default=30)
    parser.add_argument("--batch_size",type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default= "../saving_model")
    parser.add_argument("--lr", type=float, default= 0.00001)
    parser.add_argument("--gpu_mode",type=bool, default=True)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--m", type=int, default=16)

    args, _ = parser.parse_known_args()
    return args

def run_trail(config):
    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    if config.gpu_mode is True and not torch.cuda.is_available(): #虽然开启gpu模式，但是找不到GPU
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # create an instance of the model you want
    # model = Net(config)
    # model = torch.nn.DataParallel(Net(config), device_ids=[0, 1])
    model = torch.nn.DataParallel(Net(config))

    # set the logger
    log_dir = os.path.join(config.save_dir, 'logs_'+config.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # logger = Logger(log_dir)
    logger = None

    train_indices, test_indices = shuffle()
    # create your data generator
    data_train = DataGenerator(config, 'train').load_dataset()
    # create your data generator
    data_test = DataGenerator(config, 'test').load_dataset()

    # create trainer and pass all the previous components to it
    trainer = Trainer(model, config, data_train, logger, data_test)
    trainer.train_test()


    # # create tester and pass all the previous components to it
    # tester = Tester(model, config, data, logger)
    # with torch.no_grad():
    #     tester.test()
    #     # tester.test_interpolate()


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(get_params()) # convert to a dict
    params.update(tuner_params) # use dict's update method

    config = Config()
    for i in params:
        if not hasattr(config, i):
            setattr(config, i, params[i])
            print(i, params[i])

    run_trail(config)
