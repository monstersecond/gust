"""
Gain AMI and SynScore score of a network
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import dataset_utils
from analysis import k_medoids,ami_score
import analysis
import utils


def gain_ami_syn_score():
    train_dataset = dataset_utils.BindingDataset("./tmp_data", "shapes_multiobj_3_3", is_single=False, train=True, ver=True, is_ver=False)
    idx = 10
    net = torch.load('./example_net/shapes_multiobj_net_example.pty')
    device = utils.PARAM["device"]
    img_idx = 10
    test_data = train_dataset.data[img_idx]
    test_label = train_dataset.label[img_idx].reshape(-1,)
    b_s = 1
    test_data = test_data.reshape(b_s, -1).to(device)
    test_label = test_label.reshape(b_s, -1).cpu().detach().numpy()
    rfr = torch.zeros(b_s, net.input_size, device=device)
    hidden = torch.rand(b_s, net.T, net.hidden_size, device=device)
    spks, hidden, rfr, ctx, spk_all, _ = net(test_data, hidden, rfr, IT=30, device=device)
    res = []
    res2 = []
    for i in range(0, 1):
        pred, inertia = k_medoids(spk_all[i, :, :].squeeze().detach().cpu(), test_label[0], show=False, tight=True)
        ami = ami_score(test_label[0], pred.reshape(-1,))
        idx = np.where(test_label[0] > 0)[0]
        spks_tmp = spk_all[i]
        print(spks_tmp.shape)
        spks_tmp = spks_tmp[:, idx]
        spks_tmp = spks_tmp.T
        pred = np.array(pred)
        pred = pred.reshape(-1)
        spks_tmp = spks_tmp.detach().cpu().numpy()
        synchrony_score = analysis.silhouette_score(spks_tmp, pred[idx], metric=analysis.victor_purpura_metric, q=1/3)
        res.append(ami)
        res2.append(synchrony_score)
    return res, res2


if __name__ == "__main__":
    amis = []
    syns = []
    for _ in range(5):
        ami, syn = gain_ami_syn_score()
        amis = amis.append(ami)
        syns = syns.append(syn)
    print(amis)
    print(syns)
