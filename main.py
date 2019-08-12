import os
import torch
from trainers.trainer_origin_target import Trainer
# from trainers.trainer import Tester
from data_loader.data_generator import DataGenerator
from data_loader.data_generator import shuffle
from utils.utils import get_args
from utils.config import get_config_from_json
# from utils.logger import Logger
from models.model_xavier_init import Net

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
    # model = Net(config)
    model = torch.nn.DataParallel(Net(config), device_ids=[0,1])

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
    main()
