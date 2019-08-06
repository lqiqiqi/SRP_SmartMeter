import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# Plot losses
def plot_loss(config, avg_losses, show=False):

    fig, ax = plt.subplots()
    ax.set_xlim(0, config.num_epochs)
    temp = 0.0
    for i in range(len(avg_losses)):
        temp = max(np.max(avg_losses[i]), temp)
    ax.set_ylim(0, temp.detach().cpu().numpy()*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')

    if len(avg_losses) == 1:
        plt.plot(avg_losses[0], label='loss')
    else:
        plt.plot(avg_losses[0], label='G_loss')
        plt.plot(avg_losses[1], label='D_loss')
    plt.legend()

    fig_dir = config.save_dir + '/fig_' + config.exp_name
    # save figure
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    save_fn = 'Loss_values_epoch_{:d}'.format(config.num_epochs) + '.png'
    save_fn = os.path.join(config.save_dir, save_fn)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()