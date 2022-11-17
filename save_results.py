"""
This file is used to save the results to a numpy file.
The file will be used to draw figure and analyze.
"""

import torch
import numpy as np
import copy
import dataset_utils
import utils


device = utils.PARAM["device"]


def analysis_one(net_name, dataset_name):
    net = torch.load(net_name)
    train_dataset = dataset_utils.BindingDataset("./tmp_data", dataset_name, is_single=False, train=True, ver=True, is_ver=False)

    img_idx = 10

    X = train_dataset.data[img_idx] 
    X = X.reshape(X.shape[0], -1)
    label = copy.deepcopy(X)  
    X = X.to(device)
    label = label.to(device)
    IT = 30
    batch_size=X.shape[0]
    rfr = torch.zeros(batch_size, net.input_size, device=device)
    hidden = torch.rand(batch_size, net.T, net.hidden_size, device=device)

    _, hidden, rfr, _, spk_all, _ = net(X, hidden, rfr, IT=IT)

    results = {
        "spk": spk_all, 
        "label": label,
        "img": train_dataset.label[img_idx],
    }
    np.save('./results/result', results)


if __name__ == "__main__":
    analysis_one(net_name='./example_net/shapes_multiobj_net_example.pty', dataset_name="shapes_multiobj_3_3")
