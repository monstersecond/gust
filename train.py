"""
This file is used to train network and save it.
"""
from numpy.core.numeric import Inf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from dataset_utils import gain_dataset
import pbn_static
import dataset_utils
import utils


device = utils.PARAM["device"]
IMG_SIZE = utils.PARAM["img_size"]


def lr_scheduler(optimizer, shrink=0.1):
    for param_group in optimizer.param_groups:
       param_group['lr'] = param_group['lr'] * shrink
    return optimizer


def salt_pepper_noise(X, p=0.5):
    mask = torch.rand(X.shape, dtype=torch.float32)
    mask = (mask >= p)
    X = mask.to(device) * X
    return X


def entropy(spk):
    T = spk.shape[1]
    A = spk.sum(2)
    AA = A.sum(1).unsqueeze(1).repeat(1, T) + 0.00001
    B = A / AA
    loss = - B * torch.log(B + 0.00001)
    return loss.mean()

def receptive_field(spk, x):
    T = spk.shape[1]
    S = spk.shape[2]
    R = spk.mean(1)
    R = R / (R.sum(1).unsqueeze(1).repeat(1, S) + 0.000001)
    x = x / (x.sum(1).unsqueeze(1).repeat(1, S) + 0.000001)
    loss = torch.pow(x - R, 2)
    return loss.mean()


def receptive_field_sum(spk, x):
    T = spk.shape[1]
    S = spk.shape[2]
    R = spk.mean(1)
    R = R / (R.sum(1).unsqueeze(1).repeat(1, S) + 0.000001)
    x = x / (x.sum(1).unsqueeze(1).repeat(1, S) + 0.000001)
    loss = torch.pow(x - R, 2)
    return loss.sum()


def receptive_field_crosssum(spk, x):
    T = spk.shape[1]
    S = spk.shape[2]
    R = spk.mean(1)
    R = R / (R.sum(1).unsqueeze(1).repeat(1,S)+0.000001)
    loss = F.binary_cross_entropy(R,x)
    return loss.sum()



def nem_loss_still(ctx, spk, x):
    T = spk.shape[1]
    S = spk.shape[2]
    R1 = (spk.detach()*ctx).mean(1)
    R2 = ((1-spk.detach()) * ctx).mean(1)
    R1 = R1 / (R1.sum(1).unsqueeze(1).repeat(1,S)+0.000001)
    x = x / (x.sum(1).unsqueeze(1).repeat(1,S)+0.000001)
    loss1 = torch.pow(x - R1, 2)
    loss2 = torch.pow(R2, 2)
    loss = loss1 + loss2
    return loss.sum()


def nem_loss(ctx, spk, x):
    T = spk.shape[1]
    S = spk.shape[2]
    x = x.unsqueeze(1).repeat(1,T,1)
    gamma=(spk+0.000001)/(spk.sum(1).unsqueeze(1).repeat(1,T,1)+0.000001)
    loss1= -(x*(torch.log(torch.clamp(ctx,1e-6,1e6)))+(1-x)*(torch.log(torch.clamp(1-ctx,1e-6,1e6))))*gamma.detach()
    loss2 = -torch.log(torch.clamp(1-ctx,1e-6,1e6))*(1-gamma.detach())
    loss = loss1+loss2
    return loss.mean()


def nem_loss_trivel(ctx, x):
    T = ctx.shape[1]
    S = ctx.shape[2]
    x = x.unsqueeze(1).repeat(1,T,1)
    loss1 = -(x*(torch.log(torch.clamp(ctx,1e-6,1e6)))+(1-x)*(torch.log(torch.clamp(1-ctx,1e-6,1e6))))
    loss = loss1
    return loss.mean()



def nem_loss_stillmap(ctx,spk,x):
    T = spk.shape[1]
    S = spk.shape[2]
    spk = spk+0.000001
    gamma = (spk) / (spk.sum(1).unsqueeze(1).repeat(1, T, 1))
    gamma = gamma.detach()
    R1 = (gamma * ctx).sum(1)
    loss = -(x*torch.log(torch.clamp(R1, 1e-6, 1-1e-6))+(1-x)*torch.log(torch.clamp(1-R1, 1e-6, 1-1e-6)))
    return loss.mean()



def kl_loss(spk,x):
    T = spk.shape[1]
    S = spk.shape[2]
    R = spk.mean(1)
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    R = logsoftmax(R)
    x = softmax(x)
    loss = F.kl_div(R, x)
    return loss.sum()


def kl_loss2(spk,x):
    T = spk.shape[1]
    S = spk.shape[2]
    R = spk.mean(1)
    R = R / (R.sum(1).unsqueeze(1).repeat(1, S) + 0.000001)
    x = x / (x.sum(1).unsqueeze(1).repeat(1, S) + 0.000001)
    R = torch.log(torch.clamp(R, 1e-6, 1-1e-6))
    x = torch.clamp(x,1e-6,1-1e-6)
    loss = F.kl_div(R, x)
    return loss.sum()


def MSE_loss(spk, x):
    T = spk.shape[1]
    S = spk.shape[2]
    R = spk.mean(1)
    softmax = nn.Softmax(dim=1)
    R = softmax(R)
    x = softmax(x)
    loss = torch.pow(x - R, 2)
    return loss.sum()


def cross_loss(spk,x):
    T = spk.shape[1]
    S = spk.shape[2]
    R = spk.mean(1)
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    R = logsoftmax(R)
    x = softmax(x)

    loss = -x*R
    return loss.mean()


def cross_loss2(spk,x):
    T = spk.shape[1]
    S = spk.shape[2]
    R = spk.mean(1)
    R = R / (R.sum(1).unsqueeze(1).repeat(1, S) + 0.000001)
    x = x / (x.sum(1).unsqueeze(1).repeat(1, S) + 0.000001)
    R = torch.log(torch.clamp(R,1e-6,1e6))
    loss = -x*R
    return loss.mean()


def draw_spk(spk):
    print(spk.shape)
    spk = spk[0,:,:].detach().cpu()
    print(spk.shape)
    t_axis=[]
    x_axis=[]
    for i in range(spk.shape[0]):
        for j in range(spk.shape[1]):
            if spk[i,j]>0:
                t_axis.append(i)
                x_axis.append(j)
    plt.figure()
    plt.scatter(t_axis,x_axis)
    plt.show()


def draw_spk_all(spk):
    print(spk.shape)
    T = spk.shape[1]
    t_axis=[]
    x_axis=[]
    for i in range(spk.shape[0]):
        for j in range(spk.shape[1]):
            for k in range(spk.shape[2]):
                if spk[i,j,k]>0:
                    t_axis.append(i*T+j)
                    x_axis.append(k)
    plt.figure()
    plt.scatter(t_axis,x_axis)
    plt.show()


def draw_spk_img(spk,ctx,x,k,loss_para , dir_path="./tmp_img/shapes_multiobj/"):
    spk = spk[0].detach().cpu()
    ctx = ctx[0].detach().cpu()
    ctx = ctx.reshape(-1,28,28)
    img = spk.reshape(-1,28,28)
    plt.figure()
    n_row=3
    n_col = ctx.shape[0]//n_row+1
    for i in range(spk.shape[0] + 1):
        if i != spk.shape[0]:
            plt.subplot(2*n_row, n_col, i+1)
            plt.imshow(img[i, :, :])
            plt.axis('off')
            plt.subplot(2*n_row, n_col, n_row*n_col+i+1)
            plt.imshow(ctx[i, :, :])
            plt.axis('off')
        else:
            plt.subplot(2*n_row, n_col, i+1)
            plt.imshow(x[0].reshape(IMG_SIZE))
            plt.axis('off')
            plt.subplot(2*n_row, n_col, n_row*n_col+i+1)
            plt.imshow(x[0].reshape(IMG_SIZE))
            plt.axis('off')
    plt.savefig(dir_path + "/draw_spk_img_" + str(k) + ".png", dpi=500, bbox_inches="tight")
    plt.figure()
    plt.imshow(x[0].reshape(28,28))
    plt.savefig(dir_path + "/img_" + str(k) + ".png")
    

def train_pbn(net, save_dir, dataset_name, H, W, batch_size=16, num_epoch=200, lr=0.1, log_iter=False, max_unchange_epoch=20, fig_dir='./tmp_img/bars_', IT=8, p=0.65, loss_para="receptive_field_sum", tmp_net="./tmp_net/shapes_multiobj_net_"):
    """
    Train DAE network.

    Inputs:
        batch_size: batch size
        num_epoch: maximal epoch
        lr: initial learning rate
        log_iter: whether log info of each iteration in one epoch
        max_unchange_epoch: maximal epoch number unchanged verification loss, exceed will stop training 
    Outputs:
    """
    train_dataset = dataset_utils.BindingDataset("./tmp_data", dataset_name, is_single=False, train=True, ver=True, is_ver=False)
    val_dataset = dataset_utils.BindingDataset("./tmp_data", dataset_name, is_single=False, train=True, ver=True, is_ver=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    ver_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    single, multi, _, single_label, multi_label, _ = gain_dataset("./tmp_data", dataset_name)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    min_ver_loss = Inf
    unchange_epoch = 0
    LOSS = []
    AMI=[]
    LOSS_recon=[]
    Inertia=[]

    for epoch in range(num_epoch):
        running_loss = 0
        total_loss = 0
        for iter, (X, _) in enumerate(train_loader):
            X = X.reshape(X.shape[0], -1)
            label = copy.deepcopy(X)  
            X = X.to(device)
            X_r = salt_pepper_noise(X, p=p)
            label = label.to(device)
            batch_size=X.shape[0]
            rfr = torch.zeros(batch_size, net.input_size, device=device)
            hidden = torch.rand(batch_size, net.T, net.hidden_size, device=device)
            spk, hidden, rfr, ctx, spk_all, spk_pre = net(X_r, hidden, rfr, IT=IT)
            if loss_para=="receptive_field_sum":
                loss = receptive_field_sum(ctx, label)
            elif loss_para=="kl_loss":
                loss = kl_loss(ctx,label)
            elif loss_para == "kl_loss2":
                loss = kl_loss2(ctx, label)
            elif loss_para== "MSE_loss":
                loss = MSE_loss(ctx,label)
            elif loss_para=="cross_loss":
                loss = cross_loss(ctx,label)
            elif loss_para == "cross_loss2":
                loss = cross_loss2(ctx, label)
            elif loss_para=="nem_loss_trivel":
                loss = nem_loss_trivel(ctx,label)
            elif loss_para=="nem_loss_stillmap":
                loss = nem_loss_stillmap(ctx,spk,label)
            elif loss_para=="nem_loss_still":
                loss = nem_loss_still(ctx,spk,label)
            elif loss_para=="nem_loss":
                loss = nem_loss(ctx,spk_pre,label)
            running_loss += loss.cpu().item()
            total_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iter % 10 == 0) and log_iter:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'+loss_para
                    %(epoch+1, num_epoch, iter+1, len(train_dataset)//batch_size, running_loss))
                running_loss = 0
        if epoch % 10 == 0:
            torch.save(net, save_dir)
        if epoch == 60:
            optimizer = lr_scheduler(optimizer, shrink=0.5)   
        if epoch == 100:
            optimizer = lr_scheduler(optimizer, shrink=0.5) 
        if epoch == 200:
            optimizer = lr_scheduler(optimizer, shrink=0.5)      
        print("After training epoch [%d], loss [%.10f]" % (epoch, total_loss))
        rfr = torch.zeros(batch_size, net.input_size, device=device)
        hidden = torch.rand(batch_size, net.T, net.hidden_size, device=device)
        spk, hidden, rfr, ctx, spk_all, spk_pre = net(X_r, hidden, rfr, IT=IT)
        draw_spk_img(spk, ctx, X_r.cpu(), epoch, loss_para=loss_para, dir_path="./tmp_img/")

        with torch.no_grad():
            cur_ver_loss = 0
            for iter, (X, _) in enumerate(ver_loader):
                X = X.reshape(X.shape[0], -1)
                X = X.to(device)
                X_r = salt_pepper_noise(X, p=p)
                batch_size = X.shape[0]
                rfr = torch.zeros(batch_size, net.input_size, device=device)
                hidden = torch.rand(batch_size, net.T, net.hidden_size, device=device)
                spk, hidden, rfr, ctx, spk_all, spk_pre = net(X_r, hidden, rfr, IT=IT)
                if loss_para == "receptive_field_sum":
                    loss = receptive_field_sum(ctx, X)
                elif loss_para == "kl_loss":
                    loss = kl_loss(ctx, X)
                elif loss_para == "kl_loss2":
                    loss = kl_loss2(ctx, X)
                elif loss_para == "MSE_loss":
                    loss = MSE_loss(ctx, X)
                elif loss_para == "cross_loss":
                    loss = cross_loss(ctx, X)
                elif loss_para == "cross_loss2":
                    loss = cross_loss2(ctx, X)
                elif loss_para == "nem_loss_trivel":
                    loss = nem_loss_trivel(ctx, X)
                elif loss_para == "nem_loss_stillmap":
                    loss = nem_loss_stillmap(ctx, spk, X)
                elif loss_para == "nem_loss_still":
                    loss = nem_loss_still(ctx, spk, X)
                elif loss_para == "nem_loss":
                    loss = nem_loss(ctx, spk_pre, X)
                cur_ver_loss += loss.cpu().item()

            if cur_ver_loss < min_ver_loss:
                min_ver_loss = cur_ver_loss
                unchange_epoch = 0
            else:
                unchange_epoch += 1
        print("After verification epoch [%d], loss [%.5f, %.5f]" % (epoch, cur_ver_loss, min_ver_loss))
        LOSS.append(cur_ver_loss)
        if unchange_epoch > max_unchange_epoch:
            break
        torch.save(net, tmp_net + str(epoch) + ".pty")


def train():
    """
    An example uses 3 train objects
    """
    H = 28  
    W = 28  
    net = pbn_static.PBN2(H * W, 400, 10).to(device)  
    train_pbn(net, save_dir="./tmp_net/shapes_multiobj_net_train_3obj.pty", dataset_name="shapes_multiobj_3_3", H=H, W=W, log_iter=False, max_unchange_epoch=40, batch_size=1024, lr=0.01, num_epoch=400, loss_para="receptive_field_sum", p=0.65, tmp_net="./tmp_net/shapes_multiobj_net_train_3obj_")


if __name__ == "__main__":
    train()
