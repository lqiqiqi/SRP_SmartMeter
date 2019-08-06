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
            self.model.weight_init(mean=0.0, std=0.02)
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
            for iter, (input, target) in enumerate(train_data_loader):
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
    def __init__(self, model, config, data, logger):
        super(Tester, self).__init__(model, config, data, logger)

    def test(self):

        # load model
        self.model.load_model()

        # Test
        print('Test is started.')

        # load dataset
        test_data_loader = self.data

        self.DTW_loss = SoftDTWLoss()

        self.model.eval()
        loss_dtw = 0
        for input, target in test_data_loader:

            x_ = Variable(input)
            y_ = Variable(target)

            # prediction
            model_out = self.model(x_)
            loss_dtw += torch.sqrt(self.MSE_loss(model_out, y_))

        avg_dtw = loss_dtw / len(test_data_loader)

        print('avg_dtw: ', avg_dtw)

        if self.config.gpu_mode:
            self.model.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        loss = 0
        for input, target in test_data_loader:
            # input data (low resolution)
            if self.config.gpu_mode:
                x_ = Variable(input.cuda())
                y_ = Variable(target.cuda())
            else:
                x_ = Variable(input)
                y_ = Variable(target)

            # prediction
            model_out = self.model(x_)

            loss += torch.sqrt(self.MSE_loss(model_out, y_))

        avg_loss = loss / len(test_data_loader)

        print('avg_loss: ', avg_loss)
        print('Test is finished')

