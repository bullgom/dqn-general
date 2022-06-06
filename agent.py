from architecture import DQN
from parameter import HyperParameters, ETCParameters, EnvParameters
import torch
import random
import numpy as np


class Agent:

    def __init__(
        self, 
        hyp_p: HyperParameters, 
        env_p: EnvParameters,
        etc_p: ETCParameters,
        initial_checkpoint: str = ""
    ) -> None:
        
        self.hyp_p = hyp_p
        self.env_p = env_p
        self.etc_p = etc_p
        
        self.policy_net = DQN(hyp_p, env_p, etc_p).to(etc_p.device)
        self.target_net = DQN(hyp_p, env_p, etc_p).to(etc_p.device)
        self.target_net.eval()
        self.steps_done: int = 0
        
        if initial_checkpoint:
            self.load(initial_checkpoint)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:

        eps_threshold = self.hyp_p.epsilon_end + \
            (self.hyp_p.epsilon_start - self.hyp_p.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.hyp_p.epsilon_decay)

        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.env_p.output_size)]], 
                                device=self.etc_p.device, dtype=torch.long)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def step(self):
        self.steps_done += 1

    def save(self, filename: str):
        state_dict = self.policy_net.state_dict()
        torch.save(state_dict, filename)

    def load(self, filename: str):
        state_dict = torch.load(filename)
        self.policy_net.load_state_dict(state_dict)
