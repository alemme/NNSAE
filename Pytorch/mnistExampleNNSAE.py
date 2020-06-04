
import os
import argparse
import math
import NNSAE as nn
import torch
from torchvision import datasets, transforms
from plotImagesOnGrid import plotImagesOnGrid, plotPaperPlot
from numpy.matlib import zeros
import numpy as np

fileName = 'mnistExampleNNSAE.pt'

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def regression(args):
    print(args)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    batch_size = args.batch_size
    test_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])), batch_size=test_batch_size, shuffle=True, drop_last=True, **kwargs)

    numEpochs = args.epochs  # number of sweeps through data for learning
    lrateRO = args.l_rate  # learning rate for synaptic plasticity of the read-out layer
    lrateIP = args.ip_l_rate  # learning rate for intrinsic plasticity
    meanIP = args.mean_ip
    width = 28 # mnist
    inpDim = width**2
    netDim = args.hidden_dim

    # network creation
    net = nn.Nnsae(inpDim, netDim, batch_size)
    net.to(device)
    bpdc = nn.BackpropagationDecoralation(
        [net.weights], [net.h], lrateRO=lrateRO
    )
    bpdc
    loss_fkt = torch.nn.modules.MSELoss(reduction='mean')
    net.lrateRO = lrateRO
    net.lrateIP = lrateIP
    numBatches = len(train_loader)
    if not os.path.isfile(fileName) or not args.only_eval:
        # training
        for e in range(1, numEpochs+1):
            gl_loss = 0
            numSamples = 0

            for i, data in enumerate(train_loader):
                # bpdc.zero_grad()
                with torch.no_grad():
                    inp = data[0].view(data[0].shape[0], width**2).to(device)
                    numSamples += data[0].shape[0]
                    # forward path
                    out = net(inp.t()).t()
                    # calculate loss
                    loss = loss_fkt(inp, out)
                    # loss.backward()

                    # bpdc.step()
                    net.bpdc((inp-out).t())

                    # non negative constraint
                    net.weights.data[net.weights < 0] = 0
                    # intrinsic plasticity
                    net.ip()

                    # log loss
                    gl_loss += loss.item()

            # print(f'epoch ({e}\{numEpochs}) loss {gl_loss/numSamples}')
            print('epoch ({}\{}) loss {}'.format(e, numEpochs, gl_loss/numSamples))
        net.save_state_dict(fileName)
    else:
        net.load_state_dict(torch.load(fileName))
        print(net)
    ################## Evaluation ###########################
    # evaluation of basis images
    threshold = 0.1  # parameter for analysis of weights

    # sort basis images for visualization
    cnt = 0
    unused = []
    w = net.weights.t().detach().to('cpu').numpy()
    v = zeros((w.shape))
    for i in range(netDim):
        if w[i, :].max() > threshold:  # this basis image is "used"
            v[cnt, :] = w[i, :]
            cnt = cnt + 1
        else:
            unused.append(i)


    for i in range(len(unused)):
        v[cnt+i, :] = w[unused[i], :]

    print('used neurons = {}/{}'.format(cnt, netDim))


    ################## Plotting ############################
    # plotting
    numCols = 5
    if netDim >= 50:
        numCols = 10

    if not args.no_plot:
        plotImagesOnGrid(v, int(math.ceil(netDim/numCols)), numCols, width,
                        width, range(netDim), './fig/NNSAE-bars-%d-basis.png' % (netDim))

    gl_loss = 0
    numSamples = 0
    net.eval()
    orig = []
    approx = []
    activations = []
    with torch.no_grad():
        num = 0
        for i, data in enumerate(test_loader):
            inp = data[0].view(data[0].shape[0], width**2).to(device)
            label = data[1]

            numSamples += data[0].shape[0]
            # forward path
            out = net(inp.t()).t()
            # calculate loss
            loss = loss_fkt(inp, out)
            label_list = label.tolist()
            while num in label_list:
                num_index = label_list.index(num)
                orig.append(inp[num_index, :].to('cpu'))
                approx.append(out[num_index, :].to('cpu'))
                activations.append(net.h.t()[num_index, :].to('cpu'))
                num += 1

            gl_loss += loss.item()
        print('Evaluation loss MSE: {}'.format(gl_loss/numSamples))
        if not args.no_plot:
            plotPaperPlot(orig, approx, activations, width, width)


def add_model_args(parser):
    """model arguments."""

    group = parser.add_argument_group('model', 'model configurations')
    group.add_argument('-hid', '--hidden-dim', type=int, default=10,
                       metavar='N',
                       help='number of neurons in hidden layer (default: 1000')

    return parser


def add_training_args(parser):
    """Training arguments."""
    group = parser.add_argument_group('train', 'training configurations')
    group.add_argument('-lr', '--l_rate', type=float, default=0.01,
                       metavar='FLOAT', help='synaptic learning rate (default: 0.01)')
    group.add_argument('-ip_lr', '--ip_l_rate', type=float, default=0.0001,
                       metavar='FLOAT', help='intrinsic plasticity learning rate (default: 0.0001)')
    group.add_argument('-mip', '--mean_ip', type=float, default=0.2,
                       metavar='FLOAT', help='mean activity of one hidden node (default: 0.2)')
    group.add_argument('--batch-size', type=int, default=64, metavar='N',
                       help='input batch size for training (default: 64)')
    group.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                       help='number of epochs to train (default: 100)')
    return parser

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch NNSAE Model')

    group = parser.add_argument_group('setup', 'setup configurations')
    group.add_argument('--seed', type=int, default=1234,
                       metavar='N',
                       help='Random seed for reproducability.')
    group.add_argument('--cuda', action='store_true', default=False,
                       help='Run on GPU cuda accelerated')
    group.add_argument('--gpu', type=int, default=0,
                        help='used gpu cuda:?', metavar='ID')
    group.add_argument('--no-plot', action='store_true', default=False,
                       help='Show plot in interactive mode.')
    group.add_argument('--only-eval', action='store_true', default=False,
                       help='Use saved Net to reproduce plots and evaluation.')

    parser = add_model_args(parser)
    parser = add_training_args(parser)

    args = parser.parse_args()
    return args

def execute_training():
    args = get_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    regression(args)


if __name__ == "__main__":
    execute_training()