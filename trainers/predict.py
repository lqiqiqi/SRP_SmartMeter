import os
import csv
import math

import torch

import numpy as np
import pandas as pd
import torch.nn as nn

import torch.optim as optim

from torch.autograd import Variable

# from dtaidistance import dtw

# from dtw import dtw

from base.base_train import BaseTrain

from utils import utils

np.set_printoptions(threshold=1000000000)


class Tester(BaseTrain):

    def __init__(self, model, config, data_train, logger, data_test):

        super(Tester, self).__init__(model, config, data_train, logger, data_test)



    def test(self):

        # load model

        self.load_model()

        # self.load_spec_model()



        # Test

        print('Test is started.')



        # load dataset

        test_data_loader = self.data_test



        self.model.eval()



        if self.config.gpu_mode:

            self.model.cuda()

            self.MSE_loss = nn.MSELoss().cuda()

        else:

            self.MSE_loss = nn.MSELoss()



        loss_test = 0

        dtw_test = 0

        snr = 0

        flag = 0
        
        data = pd.DataFrame()

        for input_test, target_test, groundtruth in test_data_loader:

            flag += 1
            print('{} batch'.format(flag))

            if self.config.gpu_mode:

                x_test = Variable(input_test.cuda())

                y_test = Variable(groundtruth.cuda())

                y_log_test = Variable(target_test.cuda())

            else:

                x_test = Variable(input_test)

                y_test = Variable(groundtruth)

                y_log_test = Variable(target_test)

            # prediction

            model_out_test = self.model(x_test)
            
            data = pd.concat((data, pd.DataFrame(model_out_test[-1][-1].cpu().data.numpy()).T),axis=0)
            
            
            #if flag != 1:
            #   with open('../LQ_SRP_SmartMeter/predict_scale10.csv','a+') as file1:
            #       np.savetxt(file1, model_out_test.cpu().data.numpy(), delimiter=',')
            #      file1.write(',\n')
            #else:
            #   with open('../LQ_SRP_SmartMeter/predict_scale10.csv','w') as file1:
            #       np.savetxt(file1, model_out_test.cpu().data.numpy(), delimiter=',')
            #       file1.write(',\n')
        
        
        with open('../LQ_SRP_SmartMeter/predict_scale10.csv','w') as file1:
            np.savetxt(file1, data, delimiter=',')
            
        print('Test is finished')


    def load_model(self, ):

        model_dir = os.path.join(self.config.save_dir, 'model_' + self.config.exp_name)



        model_name = model_dir + '/' + self.config.model_name + '_param.pkl'  # get final model

        if os.path.exists(model_name):

            state_dict = torch.load(model_name)

            from collections import OrderedDict

            new_state_dict = OrderedDict()

            for k, v in state_dict.items():

                namekey = k[7:]  # remove `module.`

                new_state_dict[namekey] = v

            self.model.load_state_dict(new_state_dict)

            # self.model.load_state_dict(state_dict)

            print('Trained generator model is loaded.')

            return True

        else:

            print('No model exists to load.')

            self.model.weight_init()

            print('weight is initilized')

            return False
