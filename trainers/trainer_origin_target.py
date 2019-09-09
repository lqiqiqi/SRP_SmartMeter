import os
import math
# import nni
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from base.base_train import BaseTrain
from trainers.sDTW import SoftDTWLoss
from tslearn.metrics import dtw
from utils import utils
from utils.earlystopping import EarlyStopping


class Trainer(BaseTrain):
    def __init__(self, model, config, data_train, logger, data_test):
        super(Trainer, self).__init__(model, config, data_train, logger, data_test)

    def train_test(self):

        # load model if model exists weigh initialization
        if self.config.load_model is True:
            # self.load_model()
            self.load_spec_model()
        else:
            self.weight_init()


        # loss function
        if self.config.gpu_mode:
            self.model.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        # optimizer
        self.momentum = 0.9
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1.0)

        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        # scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

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
        # step = 0

        es = EarlyStopping(patience=30)

        self.model.train()  # It just sets the training mode.model.eval() to set testing mode
        for epoch in range(self.config.num_epochs):
            scheduler.step()
            epoch_loss = 0
            for iter, (input, target, groundtruth) in enumerate(train_data_loader):
                # input data (low resolution image)
                if self.config.gpu_mode:
                    x_ = Variable(input.cuda())
                    y_ = Variable(groundtruth.cuda())
                else:
                    x_ = Variable(input)
                    y_ = Variable(groundtruth)

                # scale是10的话，x_.shape is (batchsize, 1, 300)
                # scale是100的话，x_.shape is (batchsize, 1, 30)

                chop_num = 1
                slice = int(30000 / (self.config.scale_factor * chop_num)) # 改了最后的数字看要切成几份
                slice2 = int(30000 / chop_num)

                for i in range(chop_num):
                    x_1 = x_[:, :, slice*i:slice*(i+1)]
                    # update network
                    self.optimizer.zero_grad()
                    model_out = self.model(x_1)
                    loss = torch.sqrt(self.MSE_loss(model_out, y_[:, :, slice2*i:slice2*(i+1)]))
                    loss.backward()  # 结果得到是tensor
                    self.optimizer.step()
                    epoch_loss += loss

                # 注意：len(train_data_loader) 是 # train samples/batchsize，有多少个train_data_loader即需要iter多少个batch
                print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % ((epoch + 1), (iter + 1), len(train_data_loader), loss))

                # tensorboard logging
                # self.logger.scalar_summary('loss', loss, step + 1)
                # step += 1

            # avg. loss per epoch
            # 如果除以len(train_data_loader)是平均每一个batch的loss
            avg_loss.append((epoch_loss / 14000).detach().cpu().numpy())

            if (epoch + 1) % self.config.save_epochs == 0:
                self.save_model(epoch + 1)

            # caculate test loss
            with torch.no_grad():
                loss_test, _ = self.test(test_data_loader)

            # 除以测试sample个数 2000
            epoch_loss_test = loss_test / 2000

            avg_loss_test.append(float(epoch_loss_test))

            # nni.report_intermediate_result(
            #     {"default": float(epoch_loss_test), "epoch_loss": float(avg_loss[-1])})

            # if es.step(avg_loss[-1]):
            #     self.save_model(epoch=None)
            #     print('Early stop at %2d epoch' % (epoch + 1))
            #     break

            if epoch % 10 == 0 and epoch != 0:
                utils.plot_loss(self.config, [avg_loss, avg_loss_test])

        # nni.report_final_result({"default": float(avg_loss_test[-1]), "epoch_loss": float(avg_loss[-1])})

        # Plot avg. loss
        utils.plot_loss(self.config, [avg_loss, avg_loss_test])

        with torch.no_grad():
            _, dtw_test = self.test(test_data_loader, True)
            avg_dtw_test = dtw_test / len(test_data_loader)

        print('avg_loss: ', avg_loss[-1])
        print('avg_loss_log with original data: ', avg_loss_test[-1])
        print('dtw with original data: ', avg_dtw_test)
        print("Training and test is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def test(self, test_data_loader, last=False):
        loss_test = 0
        dtw_test = 0

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

            loss_test += torch.sqrt(self.MSE_loss(model_out_test, y_test))  # RMSE for re-log result and original meter data

            if last is not False:
                # dtw_test += dtw(model_out_test.squeeze(0).squeeze(0), y_test.squeeze(0).squeeze(0))
                dtw_test = 0

        return loss_test, dtw_test

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.config.save_dir, 'model_' + self.config.exp_name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(self.model.state_dict(), model_dir + '/' + self.config.model_name + '_param_epoch_%d.pkl' % epoch)
        else:  # save final model
            torch.save(self.model.state_dict(), model_dir + '/' + self.config.model_name + '_param.pkl')

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.config.save_dir, 'model_' + self.config.exp_name)

        model_name = model_dir + '/' + self.config.model_name + '_param.pkl'  # get final model
        if os.path.exists(model_name):
            state_dict = torch.load(model_name)
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     namekey = k[7:]  # remove `module.`
            #     new_state_dict[namekey] = v
            # network.load_state_dict(new_state_dict)
            self.model.load_state_dict(state_dict)
            print('Trained generator model is loaded.')
            return True
        else:
            print('No model exists to load.')
            self.weight_init()
            print('weight is initilized')
            return False

    def load_spec_model(self):
        model_dir = os.path.join(self.config.save_dir, 'model_' + self.config.exp_name)

        model_name = model_dir + '/' + self.config.model_name + '_param_epoch_120.pkl'  # get specific model
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
            self.weight_init()
            print('weight is initilized')
            return False

    def weight_init(self):
        for m in self.model.modules():
            classname = m.__class__.__name__
            if classname.find('ConvTranspose1d') != -1:
                m.weight.data = nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('Conv1d') != -1:
                m.weight.data = nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('Norm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def out_append(self, x, out):
        temp = self.model(x)
        out = torch.cat((out, temp), dim=2)
        return out





