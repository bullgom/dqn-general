from dataclasses import dataclass
import torch


@dataclass
class HyperParameters:

    target_update_interval: int
    replay_memory_size: int

    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float

    batch_size: int
    lr: float
    optimizer_type: torch.optim.Optimizer

    gamma: float
    
    num_episodes: int
    policy_update_step_interval : int
    
    image_size : int
    max_steps : int


@dataclass
class EnvParameters:
    image_w: int
    image_h: int
    output_size: int

@dataclass
class ETCParameters:

    device: torch.device

    save_interval: int
    save_path: str

    plot_length: int
    
