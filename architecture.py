import torch
import torch.nn as NN
import torch.nn.functional as F
from parameter import EnvParameters, ETCParameters, HyperParameters


class DQN(NN.Module):

    def __init__(self, hyp_param: HyperParameters, env_param: EnvParameters, etc_param: ETCParameters):
        super(DQN, self).__init__()

        self.hyp_param = hyp_param
        self.env_param = env_param
        self.etc_param = etc_param

        self.conv1 = NN.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = NN.BatchNorm2d(16)
        self.conv2 = NN.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn2 = NN.BatchNorm2d(16)
        self.conv3 = NN.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn3 = NN.BatchNorm2d(16)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        # 40 and 90 are hacks
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(hyp_param.image_size)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(hyp_param.image_size)))
        linear_input_size = convw * convh * 16
        self.head = NN.Linear(linear_input_size, env_param.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.etc_param.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.head(x)
