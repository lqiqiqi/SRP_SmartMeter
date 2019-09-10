import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tslearn.metrics import dtw
from base.base_train import BaseTrain
from utils import utils


def SNR(out, ground):
    sum = 0
    for i in range(len(ground)):
        sum += ground[i] ** 2

    noise_sum = 0
    for j in range(len(ground)):
        noise_sum += (ground[j] - out[j]) ** 2

    return 10 * math.log(sum/noise_sum ,10)

class Trainer(BaseTrain):
    def __init__(self, model, config, data, logger):
        super(Trainer, self).__init__(model, config, data, logger)

    def train(self):

        #load model if model exists weigh initialization
        if self.config.load_model is True:
            self.model.load_model()
        else:
            self.model.weight_init()
            print('weight is initilized')

        # optimizer
        self.momentum = 0.9
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.momentum)

        # loss function
        if self.config.gpu_mode:
            self.model.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.model)
        print('----------------------------------------------')

        # load dataset
        train_data_loader = self.data

        ################# Train #################
        print('Training is started.')
        avg_loss = []
        step = 0


        self.model.train() # It just sets the training mode.model.eval() to set testing mode
        for epoch in range(self.config.num_epochs):

            epoch_loss = 0
            for iter, (input, target, _) in enumerate(train_data_loader):
                # input data (low resolution image)
                if self.config.gpu_mode:
                    x_ = Variable(input.cuda())
                    y_ = Variable(target.cuda())
                else:
                    x_ = Variable(input)
                    y_ = Variable(target)

                # update network
                self.optimizer.zero_grad()
                model_out = self.model(x_)
                loss = torch.sqrt(self.MSE_loss(model_out, y_))
                loss.backward()
                self.optimizer.step()

                # log
                epoch_loss += loss
                print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % ((epoch + 1), (iter + 1), len(train_data_loader), loss))

                # tensorboard logging
                # self.logger.scalar_summary('loss', loss, step + 1)
                step += 1

            # avg. loss per epoch
            avg_loss.append(epoch_loss / len(train_data_loader))

            if (epoch + 1) % self.config.save_epochs == 0:
                self.model.save_model(epoch + 1)

        # Plot avg. loss
        utils.plot_loss(self.config, [avg_loss])
        print('avg_loss: ', avg_loss[-1])
        print("Training is finished.")

        # Save final trained parameters of model
        self.model.save_model(epoch=None)


class Tester(BaseTrain):
    def __init__(self, model, config, data_train, logger, data_test):
        super(Tester, self).__init__(model, config, data_train, logger, data_test)

    def test(self):

        # load model
        # self.load_model()
        self.load_spec_model()

        # self.model.load_state_dict(torch.load('SRPResNet_100-1000_64ndf.pth'))

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

        for input_test, target_test, groundtruth in test_data_loader:
            # input data (low resolution)
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

            # out_postlog = torch.div(torch.exp(model_out_test * math.log(100)) - 1.0, 1000.)
            #
            loss_test += torch.sqrt(
                self.MSE_loss(model_out_test, y_test))  # RMSE for re-log result and original meter data

            for sample in range(y_test.size()[0]):
                snr += SNR(model_out_test[sample][-1], y_test[sample][-1])

            dtw_one_sample = 0
            # print(y_test.size())
            print(flag)
            flag += 1

            # # print(y_test.size()) torch.Size([32, 1, 30000])
            # # print(model_out_test.size()) torch.Size([32, 1, 30000])
            # for sample in range(y_test.size()[0]):
            #     for i in range(0, y_test.size()[-1], 100):
            #         if i+99 < y_test.size()[-1]:
            #             temp_dtw = dtw(model_out_test[sample][-1][i:i+99], y_test[sample][-1][i:i+99])
            #             print(temp_dtw)
            #             dtw_one_sample += temp_dtw
            #         else:
            #             print(i)
            #             break
            #
            # # dtw_test += dtw_one_sample / (len(y_test.squeeze(0).squeeze(0)) - 100 + 1)
            # # print(dtw_one_sample / (len(y_test.squeeze(0).squeeze(0)) - 100 + 1))
            # dtw_test += dtw_one_sample / 300
            # print(dtw_one_sample / (300*32))

        snr_avg = snr / 2000
        avg_loss = loss_test / len(test_data_loader)
        avg_dtw_test = dtw_test / len(test_data_loader)

        print("average SNR: ", snr_avg)
        print('avg_loss with original data: ', avg_loss)
        print('avg_dtw_test with original data: ', avg_dtw_test)
        print('Test is finished')

    def load_spec_model(self):
        model_dir = os.path.join(self.config.save_dir, 'model_' + self.config.exp_name)

        model_name = model_dir + '/' + self.config.model_name + '_param_epoch_30.pkl'  # get specific model
        if os.path.exists(model_name):
            state_dict = torch.load(model_name)
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     namekey = k[7:]  # remove `module.`
            #     new_state_dict[namekey] = v
            # self.model.load_state_dict(new_state_dict)
            self.model.load_state_dict(state_dict)
            print('Trained generator model is loaded.')
            return True
        else:
            print('No model exists to load.')
            self.model.weight_init()
            print('weight is initilized')
            return False

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
            print('Trained generator model is loaded.')
            return True
        else:
            print('No model exists to load.')
            self.model.weight_init()
            print('weight is initilized')
            return False