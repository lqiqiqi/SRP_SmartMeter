import os
import torch
from trainers.trainer_plot_test import Trainer
from trainers.trainer import Tester
from data_loader.data_generator import DataGenerator
from data_loader.data_generator import shuffle
from utils.utils import get_args
from utils.config import get_config_from_json
# from utils.logger import Logger
from models.model_bicubic import Net

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    args = get_args()
    config, _ = get_config_from_json(args.config)
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    if config.gpu_mode is True and not torch.cuda.is_available(): #虽然开启gpu模式，但是找不到GPU
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # create an instance of the model you want
    model = Net(config)

    # set the logger
    log_dir = os.path.join(config.save_dir, 'logs_'+config.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # logger = Logger(log_dir)
    logger = None

    # create your data generator
    # data_train = DataGenerator(config, 'debug').load_dataset()
    # create your data generator
    data_test = DataGenerator(config, 'test').load_dataset()
    data_train = None

    # create trainer and pass all the previous components to it
    # trainer = Trainer(model, config, data_train, logger, data_test)
    # trainer.train_test()


    # # create tester and pass all the previous components to it
    # 使用最后一个模型：在trainer.py中使用load_model函数
    # 使用非最后一个模型：在base_model模块中指定特定模型，并在trainer.py中使用load_spec_model函数
    tester = Tester(model, config, data_train, logger, data_test)
    with torch.no_grad():
        tester.test()
    #     # tester.test_interpolate()


if __name__ == '__main__':
    main()
