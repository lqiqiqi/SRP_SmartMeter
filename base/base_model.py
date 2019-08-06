import os
import torch


class BaseModel:
    def __init__(self, config):
        self.config = config

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.config.save_dir, 'model_'+ self.config.exp_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(self.state_dict(), model_dir + '/' + self.config.model_name + '_param_epoch_%d.pkl' % epoch)
        else: # save final model
            torch.save(self.state_dict(), model_dir + '/' + self.config.model_name + '_param.pkl')

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.config.save_dir, 'model_'+ self.config.exp_name)

        model_name = model_dir + '/' + self.config.model_name + '_param.pkl' # get final model
        if os.path.exists(model_name):
            self.load_state_dict(torch.load(model_name))
            print('Trained generator model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False


