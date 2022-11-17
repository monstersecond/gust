"""
The analysis functions
"""
from py import process
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
import copy
from FCA import victor_purpura_metric
from torch.utils.data import DataLoader
import dataset_utils
import os
import utils
import threading
import multiprocessing


os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
devices = []
DEVICE_NUM = 3
for i in range(DEVICE_NUM):
    devices.append(torch.device("cuda:"+str(i) if torch.cuda.is_available() else "cpu"))

device = utils.PARAM["device"]


def k_medoids(spk_to_copied, label, show=True, tight=False, back=10, q=3):
    spk = copy.deepcopy(spk_to_copied)
    if not tight:
        spk = spk[-back:, :, :, :]
        sizex = spk.shape[2]
        sizey = spk.shape[3]
        spk = spk.reshape(-1, sizex, sizey)
    else:
        spk = spk.reshape(-1, 28, 28)
    spk_tmp = copy.deepcopy(spk)
    spk = spk_tmp
    K = np.max(label) + 1
    spk = spk.reshape(spk.shape[0], -1)
    spk = spk.T 
    estimator = KMedoids(n_clusters=int(K),init = 'k-medoids++',metric=victor_purpura_metric)
    estimator.fit(spk)
    label_pred = estimator.labels_
    label_pred = label_pred.reshape(28, 28)
    inertia=estimator.inertia_
    if show == True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(label_pred)
        plt.subplot(1, 2, 2)
        plt.imshow(label)
        plt.show()
    return label_pred, inertia


def ami_score(true_groups, predicted):
    idxs = np.where(true_groups != 0.0)[0]
    print(idxs)
    print(true_groups[idxs].shape)
    print(predicted[idxs].shape)
    score = adjusted_mutual_info_score(true_groups[idxs], predicted[idxs])
    return score


def draw_spk_img(spk,ctx,x,k, dir_path,mode):
    if mode == "static":
        spk = spk[0].detach().cpu()
        ctx = ctx[0].detach().cpu()
        ctx = ctx.reshape(-1,28,28)
        img = spk.reshape(-1,28,28)
        plt.figure()
        n_row=3
        n_col = ctx.shape[0]//n_row+1
        for i in range(spk.shape[0]):
            plt.subplot(2*n_row, n_col, i+1)
            plt.imshow(img[i, :, :])
            plt.axis('off')
            plt.subplot(2*n_row, n_col, n_row*n_col+i+1)
            plt.imshow(ctx[i, :, :])
            plt.axis('off')
        plt.savefig(dir_path+"draw_spk_img"+str(k)+".png", dpi=500, bbox_inches="tight")
        plt.figure()
        plt.imshow(x[0].reshape(28,28))
        plt.savefig(dir_path+"img" + str(k) + ".png")
        
    if mode == "move":
        spk = spk[0].detach().cpu()
        ctx = ctx[0].detach().cpu()
        ctx = ctx.reshape(-1, 28, 28)
        img = spk.reshape(-1, 28, 28)
        plt.figure()
        n_row = 3
        n_col = ctx.shape[0] // n_row + 1
        for i in range(spk.shape[0]):
            plt.subplot(2 * n_row, n_col, i + 1)
            plt.imshow(img[i, :, :])
            plt.axis('off')
            plt.subplot(2 * n_row, n_col, n_row * n_col + i + 1)
            plt.imshow(ctx[i, :, :])
            plt.axis('off')
        plt.savefig(dir_path + "draw_spk_img" + str(k) + ".png", dpi=500, bbox_inches="tight")
        plt.figure()
        plt.imshow(x[0].reshape(28, 28))
        plt.savefig(dir_path + "img" + str(k) + ".png")
        

class GainAmiKernel(threading.Thread):
    def __init__(self, thread_id, net, iteration_step, test_loader, device, batch_size=1024):
        super(GainAmiKernel, self).__init__()
        self.thread_id = thread_id
        self.net = copy.deepcopy(net)
        self.net = self.net.to(device)
        self.device = device
        self.iteration_step = iteration_step
        self.batch_size = batch_size
        self.res = []
        self.test_loader = test_loader

    def run(self):
        self.res = []
        for iter, (test_data, test_label) in enumerate(self.test_loader):
            b_s = test_data.shape[0]
            test_data = test_data.reshape(b_s, -1).to(self.device)
            test_label = test_label.reshape(b_s, -1).cpu().detach().numpy()
            rfr = torch.zeros(b_s, self.net.input_size, device=self.device)
            hidden = torch.rand(b_s, self.net.T, self.net.hidden_size, device=self.device)
            spks, hidden, rfr, ctx, spk_all = self.net(test_data, hidden, rfr, IT=self.iteration_step, device=self.device)
            for i in range(spks.shape[0]):
                print("thread %d, ITERATION %d, i %d" % (self.thread_id, iter, i))
                pred, inertia = k_medoids(spks[i].squeeze().detach().cpu(), test_label[i], show=False, tight=True)
                ami = ami_score(test_label[i], pred.reshape(-1,))
                self.res.append(ami)

    def get_result(self):
        return self.res


def gain_ami(net, iteration_step, dataset_name, thread_num=20, batch_size=1024, data_num=1000):
    net = net.cpu()
    test_dataset = dataset_utils.BindingDataset("./tmp_data", dataset_name, train=False)
    test_dataset.data = test_dataset.data[:data_num]
    test_dataset.label = test_dataset.label[:data_num]
    datanum = test_dataset.data.shape[0]  
    data_size = datanum // thread_num
    data_sizes = [0]
    for i in range(thread_num - 1):
        data_sizes.append(data_size)
    data_sizes.append(datanum - data_size * (thread_num - 1))
    for i in range(1, thread_num + 1):
        data_sizes[i] += data_sizes[i - 1]
    threads = []
    for i in range(thread_num):
        start = data_sizes[i]
        end = data_sizes[i + 1]
        t_test_dataset = dataset_utils.BindingDatasetThread(test_dataset.data[start:end], test_dataset.label[start:end])
        test_loader = DataLoader(dataset=t_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        thread = GainAmiKernel(i, net, iteration_step, test_loader, devices[i % DEVICE_NUM], batch_size=batch_size)
        threads.append(thread)
        thread.start()
    
    for i in range(thread_num):
        threads[i].join()
    
    ami = []
    for i in range(thread_num):
        ami.extend(threads[i].get_result())
    
    return ami


def gain_ami_process_kernel(process_id, test_loader, net, iteration_step, device, return_dict):
    net = net.to(device)
    res = []
    res2 = []
    for iter, (test_data, test_label) in enumerate(test_loader):
        b_s = test_data.shape[0]
        test_data = test_data.reshape(b_s, -1).to(device)
        test_label = test_label.reshape(b_s, -1).cpu().detach().numpy()
        rfr = torch.zeros(b_s, net.input_size, device=device)
        hidden = torch.rand(b_s, net.T, net.hidden_size, device=device)
        spks, hidden, rfr, ctx, spk_all, _ = net(test_data, hidden, rfr, IT=iteration_step, device=device)
        for i in range(spks.shape[0]):
            pred, inertia = k_medoids(spks[i].squeeze().detach().cpu(), test_label[i], show=False, tight=True)
            ami = ami_score(test_label[i], pred.reshape(-1,))
            idx = np.where(test_label[i] > 0)[0]
            spks_tmp = spks[i]
            spks_tmp = spks_tmp[:, idx]
            spks_tmp = spks_tmp.T
            pred = np.array(pred)
            pred = pred.reshape(-1)
            spks_tmp = spks_tmp.detach().cpu().numpy()
            if np.max(pred[idx]) - np.min(pred[idx]) == 0:
                res.append(ami)
            else:
                synchrony_score = silhouette_score(spks_tmp, pred[idx], metric=victor_purpura_metric, q=1/3)
                
                res.append(ami)
                res2.append(synchrony_score)

    return_dict[process_id] = [res, res2]


def gain_ami_process(net, iteration_step, dataset_name, thread_num=20, batch_size=1024, data_num=1000):
    try:
        multiprocessing.set_start_method('spawn')
    except:
        pass
    net = net.cpu()
    test_dataset = dataset_utils.BindingDataset("./tmp_data", dataset_name, train=False)
    test_dataset.data = test_dataset.data[:data_num]
    test_dataset.label = test_dataset.label[:data_num]
    datanum = test_dataset.data.shape[0]  
    data_size = datanum // thread_num
    data_sizes = [0]
    for i in range(thread_num - 1):
        data_sizes.append(data_size)
    data_sizes.append(datanum - data_size * (thread_num - 1))
    for i in range(1, thread_num + 1):
        data_sizes[i] += data_sizes[i - 1]
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i in range(thread_num):
        start = data_sizes[i]
        end = data_sizes[i + 1]
        t_test_dataset = dataset_utils.BindingDatasetThread(test_dataset.data[start:end], test_dataset.label[start:end])
        test_loader = DataLoader(dataset=t_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        net_proc = copy.deepcopy(net)
        process = multiprocessing.Process(target=gain_ami_process_kernel, args=(i, test_loader, net_proc, iteration_step, devices[i % DEVICE_NUM], return_dict))
        processes.append(process)
        process.start()
    for i in range(thread_num):
        processes[i].join()
    ami = []
    syn = []
    for i in range(thread_num):
        ami.extend(return_dict[i][0])
        syn.extend(return_dict[i][1])
    return ami, syn


def draw_spk_img(spk,ctx,x,k, dir_path="./tmp_img/shapes_multiobj/", IMG_SIZE=(28, 28)):
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
    plt.savefig(dir_path+"draw_spk_img"+str(k)+".png", dpi=500, bbox_inches="tight")
    plt.figure()
    plt.imshow(x[0].reshape(28,28))
    plt.savefig(dir_path+"img" + str(k) + ".png")

def gain_reconstruct_result(net, iteration_step, test_data, test_label, iter):
    net = net.to(device)
    test_data = test_data.reshape(1, -1).to(device)
    test_label = test_label.reshape(1, -1).cpu().detach().numpy()
    rfr = torch.zeros(1, net.input_size, device=device)
    hidden = torch.rand(1, net.T, net.hidden_size, device=device)
    spk, hidden, rfr, ctx, spk_all, _ = net(test_data, hidden, rfr, IT=iteration_step, device=device)
    draw_spk_img(spk, ctx, test_label, iter, dir_path="./tmp_img/shapes_multiobj_reconstruct_result/")
