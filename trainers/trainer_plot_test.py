import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from base.base_train import BaseTrain
from trainers.sDTW import SoftDTWLoss
from utils import utils


class Trainer(BaseTrain):
    def __init__(self, model, config, data_train, logger, data_test):
        super(Trainer, self).__init__(model, config, data_train, logger, data_test)

    def train_test(self):

        #load model if model exists weigh initialization
        if self.config.load_model is True:
            self.model.load_model()
        else:
            self.model.weight_init()
            print('weight is initilized')

        # optimizer
        self.momentum = 0.9
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.momentum, weight_decay=1.0)

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
        train_data_loader = self.data_train
        test_data_loader = self.data_test

        ################# Train #################
        print('Training is started.')
        avg_loss = []
        avg_loss_test = []
        avg_loss_log_test = []
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
                loss.backward() # 结果得到是tensor
                self.optimizer.step()

                # log
                epoch_loss += loss
                print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % ((epoch + 1), (iter + 1), len(train_data_loader), loss))

                # tensorboard logging
                self.logger.scalar_summary('loss', loss, step + 1)
                step += 1

            # avg. loss per epoch
            avg_loss.append((epoch_loss / len(train_data_loader)).detach().cpu().numpy())

            if (epoch + 1) % self.config.save_epochs == 0:
                self.model.save_model(epoch + 1)

            # caculate test loss
            with torch.no_grad():
                loss_test = 0
                loss_log_test = 0
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
                    relog = torch.mul(torch.add(torch.exp(torch.mul(model_out_test, math.log(100))), -1), 1 / 100)

                    loss_test += torch.sqrt(self.MSE_loss(relog, y_test))  # RMSE for re-log result and original meter data
                    loss_log_test += torch.sqrt(self.MSE_loss(model_out_test, y_log_test))  # RMSE for log result # 结果得到是np.float

                epoch_loss_test = loss_test / len(test_data_loader)
                epoch_loss_log_test = loss_log_test / len(test_data_loader)

                avg_loss_test.append(float(epoch_loss_test))
                avg_loss_log_test.append(float(epoch_loss_log_test))


        # Plot avg. loss
        utils.plot_loss(self.config, [avg_loss, avg_loss_log_test])
        utils.plot_loss(self.config, [avg_loss_test], True)

        print('avg_loss: ', avg_loss[-1])
        print('avg_loss_log with original data: ', avg_loss_test[-1])
        print('avg_loss_log with log data: ', avg_loss_log_test[-1])
        print("Training and test is finished.")

        # Save final trained parameters of model
        self.model.save_model(epoch=None)




