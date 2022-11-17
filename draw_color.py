"""
Draw the dyeing results like the first line of Fig. 4
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
from matplotlib.colors import hsv_to_rgb
import dataset_utils


device = utils.PARAM["device"]
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 


def coloring(spk, choose, para=[29, 9], idx = [0], show=True): 
    iteration = spk.shape[0]
    delay = spk.shape[1]  
    spk = spk.transpose(0, 2, 3, 1)   
    T = spk.shape[-1]
    w = para[0] / para[1] / 2
    colors = 0.25 * (np.sin(2 * w * np.pi * np.linspace(0, 1, T, endpoint=False) + 0.2) + 0) + 0.75 
    colors = np.tile(colors, (spk.shape[0], spk.shape[1], spk.shape[2], 1))
    results = spk * colors
    if results.shape[-1] != 3:
        nr_colors = results.shape[-1]
        hsv_colors = np.ones((nr_colors, 3))
        hsv_colors[:, 0] = (np.linspace(0, 1, nr_colors, endpoint=False) + 2 / 3) % 1.0
        color_conv = hsv_to_rgb(hsv_colors)
        results = results.reshape(-1, nr_colors).dot(color_conv).reshape(results.shape[:-1] + (3,))
    if show:
        plt.figure()
        for i in range(len(idx)):
            plt.imshow(results[idx[i], :, :, :], interpolation='nearest')
            plt.axis('off')
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['figure.dpi'] = 300
            plt.savefig('./paper_figure/coloring_iter_' + str(choose) + '.png', dpi=500, bbox_inches='tight', pad_inches=0)

    return results

def coloring_moving(spk,x, para=[20, 9], idx = [0], show=True):
    iteration = spk.shape[0]
    delay = spk.shape[1]
    x = x.reshape(15,10, 28, 28).cpu()
    print("xshape",x.shape)
    spk = spk.transpose(0, 2, 3, 1)
    T = spk.shape[-1]
    w = para[0] / para[1]
    colors = 0.5 * (np.sin(2 * w * np.pi * np.linspace(0, 1, T, endpoint=False) + 0.25) + 1) 
    colors = np.tile(colors, (spk.shape[0], spk.shape[1], spk.shape[2], 1))
    results = spk * colors
    if results.shape[-1] != 3:
        nr_colors = results.shape[-1]
        hsv_colors = np.ones((nr_colors, 3))
        hsv_colors[:, 0] = (np.linspace(0.2, 0.8, nr_colors, endpoint=False) + 2 / 3) % 1.0
        color_conv = hsv_to_rgb(hsv_colors)
        results = results.reshape(-1, nr_colors).dot(color_conv).reshape(results.shape[:-1] + (3,))
    if show:
        plt.figure()
        plt.subplots_adjust(wspace=0.1, hspace=0.05)  
        for i in range(len(idx)):
            plt.subplot(2, 15, i + 1)
            plt.imshow(results[idx[i], :, :, :], interpolation='nearest')
            plt.axis('off')
            plt.subplot(2, 15, 15+i + 1)
            plt.imshow(x[idx[i], :, :, :].mean(0), interpolation='nearest',cmap='gray')
            plt.axis('off')
        plt.savefig('./paper_figure/MMoving_coloring_' + ".png")
    return results


def draw_spk_img_alltime(spk,ctx,x,dir_path="./paper_figure/"):
    ctx = ctx.reshape(-1,28,28)
    img = spk.reshape(-1,28,28)
    x = x.reshape(-1,28,28).cpu()
    plt.figure()
    n_col=20
    n_row = ctx.shape[0]//n_col+1
    for i in range(img.shape[0]):
        plt.subplot(3*n_row, n_col, i+1)
        plt.imshow(img[i, :, :])
        plt.axis('off')
        plt.subplot(3*n_row, n_col, n_row*n_col+i+1)
        plt.imshow(ctx[i, :, :])
        plt.axis('off')
        plt.subplot(3 * n_row, n_col, 2*n_row * n_col + i + 1)
        plt.imshow(x[i, :, :])
        plt.axis('off')
    plt.savefig(dir_path+"draw_spk_img_alltime.png", dpi=100, bbox_inches="tight")
    plt.figure()


if __name__ == "__main__":
    # 1. load dataset
    train_dataset = dataset_utils.BindingDataset("./tmp_data", "shapes_multiobj_3_3", is_single=False, train=True, ver=True, is_ver=False)
    idx = 10
    
    # 2. load trained network
    net = torch.load('./example_net/shapes_multiobj_net_example.pty')

    # 3. gain binding results
    x = np.array(train_dataset.data[idx], dtype=np.float32)
    x = torch.tensor(x, device=device)
    x = x.unsqueeze(0)
    x = x.reshape(x.shape[0], -1)
    label = train_dataset.label[idx]
    rfr = torch.zeros(1, net.input_size, device=device)
    hidden = torch.rand(1, net.T, net.hidden_size, device=device)
    spk, hidden, rfr, ctx, spk_all, spk_pre = net(x, hidden, rfr, IT=30)

    # 4. choose iteration to draw 
    for i in [0, 5, 10, 15, 20, 25, 29]:
        coloring(spk_all[i].reshape(1, spk.shape[1], 28, 28).detach().cpu().numpy()[-2:, :, :, :], choose=i)
