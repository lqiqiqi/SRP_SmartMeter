import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from base.base_train import BaseTrain
from trainers.sDTW import SoftDTWLoss
from utils import utils


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
                self.logger.scalar_summary('loss', loss, step + 1)
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
        # self.model.load_model()
        self.model.load_spec_model()

        # Test
        print('Test is started.')

        # load dataset
        test_data_loader = self.data_test

        self.model.eval()

        # self.DTW_loss = SoftDTWLoss()
        #
        # loss_dtw = 0
        # for input, target, groundtruth in test_data_loader:
        #
        #     x_ = Variable(input)
        #     y_ = Variable(target)
        #
        #     # prediction
        #     model_out = self.model(x_)
        #     loss_dtw += self.DTW_loss(model_out, y_)
        #
        # avg_dtw = loss_dtw / len(test_data_loader)
        #
        # print('avg_dtw: ', avg_dtw)

        if self.config.gpu_mode:
            self.model.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        loss = 0
        loss_log = 0
        for input, target, groundtruth in test_data_loader:
            # input data (low resolution)
            if self.config.gpu_mode:
                x_ = Variable(input.cuda())
                y_ = Variable(groundtruth.cuda())
                y_log = Variable(target.cuda())
            else:
                x_ = Variable(input)
                y_ = Variable(groundtruth)
                y_log = Variable(target)

            # prediction
            model_out = self.model(x_)
            relog = torch.mul(torch.add(torch.exp(torch.mul(model_out, math.log(100))), -1), 1/100)

            loss += torch.sqrt(self.MSE_loss(relog, y_)) # RMSE for re-log result and original meter data
            loss_log += torch.sqrt(self.MSE_loss(model_out, y_log)) # RMSE for log result


        avg_loss = loss / len(test_data_loader)
        avg_loss_log = loss_log / len(test_data_loader)

        print('avg_loss with original data: ', avg_loss)
        print('avg_loss_log with log data: ', avg_loss_log)
        print('Test is finished')


    def test_interpolate(self):

        # Test
        print('Test is started.')

        # load dataset
        test_data_loader = self.data

        if self.config.gpu_mode:
            self.model.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        loss_linear = 0
        loss_bicubic = 0
        for input, target, groundtruth in test_data_loader:
            # input data (low resolution)
            if self.config.gpu_mode:
                x_ = Variable(input.cuda())
                y_ = Variable(groundtruth.cuda())
                y_log = Variable(target.cuda())
            else:
                x_ = Variable(input)
                y_ = Variable(groundtruth)
                y_log = Variable(target)

            # prediction
            relog = torch.mul(torch.add(torch.exp(torch.mul(x_, math.log(100))), -1), 1 / 100) # 恢复low resolution data
            print(relog.size())
            interp_out_linear = torch.nn.functional.interpolate(relog, scale_factor=self.config.scale_factor,
                                                                mode = 'linear')
            # interp_out_bicubic = torch.nn.functional.interpolate(relog.unsqueeze(0), scale_factor=self.config.scale_factor,
            #                                                     mode='bicubic')

            loss_linear += torch.sqrt(self.MSE_loss(interp_out_linear, y_)) # RMSE for re-log result and original meter data
            # loss_bicubic += torch.sqrt(self.MSE_loss(interp_out_bicubic, y_))  # RMSE for re-log result and original meter data


        avg_loss_linear = loss_linear / len(test_data_loader)
        # avg_loss_bicubic = loss_bicubic / len(test_data_loader)

        print('avg_loss with linear: ', avg_loss_linear)
        # print('avg_loss with bibcubic: ', avg_loss_bicubic)
        print('Test is finished')

