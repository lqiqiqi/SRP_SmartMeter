import os
import torch
from trainers.trainer import Trainer
from data_loader.data_generator import DataGenerator
from utils.utils import get_args
from utils.config import get_config_from_json
from utils.logger import Logger
from models.model import Net

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    # try:
    args = get_args()
    config, _ = get_config_from_json(args.config)
    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    # if config.gpu_mode and not torch.cuda.is_available():
        # raise Exception("No GPU found, please run without --gpu_mode=False")

    # create your data generator
    data = DataGenerator(config, 'debug')

    # create an instance of the model you want
    model = Net(config)
    #load model if exists
    model.load_model()

    # set the logger
    log_dir = os.path.join(config.save_dir, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = Logger(log_dir)

    # create trainer and pass all the previous components to it
    trainer = Trainer(model, config, data, logger)
    trainer.train()


if __name__ == '__main__':
    main()
