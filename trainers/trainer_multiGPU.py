import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from base.base_train import BaseTrain
from trainers.sDTW import SoftDTWLoss
from utils import utils
from utils.earlystopping import EarlyStopping


class Trainer(BaseTrain):
    def __init__(self, model, config, data_train, logger, data_test):
        super(Trainer, self).__init__(model, config, data_train, logger, data_test)

    def train_test(self):

        #load model if model exists weigh initialization
        if self.config.load_model is True:
            # self.load_model(self.model)
            self.load_spec_model(self.model)
        else:
            try:
                self.model.weight_init()
            except:
                for m in self.model.modules():
                    classname = m.__class__.__name__
                    if classname.find('Linear') != -1:
                        torch.nn.init.kaiming_normal(m.weight)
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif classname.find('Conv2d') != -1:
                        torch.nn.init.kaiming_normal(m.weight)
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif classname.find('ConvTranspose2d') != -1:
                        torch.nn.init.kaiming_normal(m.weight)
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif classname.find('Norm') != -1:
                        m.weight.data.normal_(1.0, 0.02)
                        if m.bias is not None:
                            m.bias.data.zero_()

        # optimizer
        self.momentum = 0.9
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1.0)

        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.01)
        # scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

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

        es = EarlyStopping(patience=8)

        self.model.train() # It just sets the training mode.model.eval() to set testing mode
        for epoch in range(self.config.num_epochs):
            scheduler.step()
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
                self.save_model(self.model, epoch + 1)

            # caculate test loss
            with torch.no_grad():
                loss_test, loss_log_test = self.test(test_data_loader)

            epoch_loss_test = loss_test / len(test_data_loader)
            epoch_loss_log_test = loss_log_test / len(test_data_loader)

            avg_loss_test.append(float(epoch_loss_test))
            avg_loss_log_test.append(float(epoch_loss_log_test))

            if es.step(float(epoch_loss_test)):
                self.save_model(epoch=None)
                print('Early stop at %2d epoch' % (epoch + 1))
                break

        # Plot avg. loss
        utils.plot_loss(self.config, [avg_loss, avg_loss_log_test])
        utils.plot_loss(self.config, [avg_loss_test], origin=True)

        print('avg_loss: ', avg_loss[-1])
        print('avg_loss_log with original data: ', avg_loss_test[-1])
        print('avg_loss_log with log data: ', avg_loss_log_test[-1])
        print("Training and test is finished.")

        # Save final trained parameters of model
        self.save_model(self.model, epoch=None)

    def test(self, test_data_loader):
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

        return loss_test, loss_log_test

    def save_model(self, network, epoch=None):
        model_dir = os.path.join(self.config.save_dir, 'model_'+ self.config.exp_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(network.state_dict(), model_dir + '/' + self.config.model_name + '_param_epoch_%d.pkl' % epoch)
        else: # save final model
            torch.save(network.state_dict(), model_dir + '/' + self.config.model_name + '_param.pkl')

        print('Trained model is saved.')

    def load_model(self, network):
        model_dir = os.path.join(self.config.save_dir, 'model_'+ self.config.exp_name)

        model_name = model_dir + '/' + self.config.model_name + '_param.pkl' # get final model
        if os.path.exists(model_name):
            state_dict = torch.load(model_name)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                namekey = k[7:]  # remove `module.`
                new_state_dict[namekey] = v
            network.load_state_dict(new_state_dict)
            print('Trained generator model is loaded.')
            return True
        else:
            print('No model exists to load.')
            network.weight_init()
            print('weight is initilized')
            return False

    def load_spec_model(self, network):
        model_dir = os.path.join(self.config.save_dir, 'model_'+ self.config.exp_name)

        model_name = model_dir + '/' + self.config.model_name + '_param_epoch_60.pkl' # get specific model
        if os.path.exists(model_name):
            state_dict = torch.load(model_name)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                namekey = k[7:]  # remove `module.`
                new_state_dict[namekey] = v
            network.load_state_dict(new_state_dict)
            print('Trained generator model is loaded.')
            return True
        else:
            print('No model exists to load.')
            network.weight_init()
            print('weight is initilized')
            return False




