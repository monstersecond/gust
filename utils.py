"""
Save global parameters
"""
import torch
import torch.nn

PARAM = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "img_size": (28, 28)
}
