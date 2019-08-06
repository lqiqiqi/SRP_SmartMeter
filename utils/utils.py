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
def plot_loss(avg_losses, num_epochs, save_dir='', show=False):
    avg_losses_np = avg_losses.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    temp = 0.0
    for i in range(len(avg_losses_np)):
        temp = max(np.max(avg_losses_np[i]), temp)
    ax.set_ylim(0, temp*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')

    if len(avg_losses_np) == 1:
        plt.plot(avg_losses_np[0], label='loss')
    else:
        plt.plot(avg_losses_np[0], label='G_loss')
        plt.plot(avg_losses_np[1], label='D_loss')
    plt.legend()

    # save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
    save_fn = os.path.join(save_dir, save_fn)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()