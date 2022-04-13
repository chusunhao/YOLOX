import torch
from yolox.exp import get_exp

if __name__ == "__main__":
    exp_file = "exps/example/custom/tph-yolox_x.py"
    exp = get_exp(exp_file)
    exp.get_model()

    inputs = torch.rand(2, 640, 640, 3)
    outputs = exp.model(inputs)
