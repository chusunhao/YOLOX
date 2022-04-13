import torch
from yolox.exp import get_exp

if __name__ == "__main__":
    # exp_file = "exps/example/custom/tph-yolox_x.py"
    exp_file = "exps/default/yolox_l.py"
    exp = get_exp(exp_file=exp_file, exp_name=None)
    exp.get_model()

    model = exp.model
    model.training = False

    inputs = torch.rand(2, 3, 640, 640)
    outputs = model(inputs)
    print("done")
