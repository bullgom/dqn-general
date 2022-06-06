from parameter import HyperParameters, ETCParameters
from agent import Agent
from replay_memory import ReplayMemory
from transition import Transition
import torch
import torch.nn.functional as F

class Teacher:

    def __init__(
        self, 
        hyp_p: HyperParameters, 
        etc_p: ETCParameters,
        agent: Agent, 
        mem: ReplayMemory,
    ) -> None:
        self.hyp_p = hyp_p
        self.etc_p = etc_p
        self.agent = agent
        self.mem = mem
        self.optim = hyp_p.optimizer_type(agent.policy_net.parameters(), lr=hyp_p.lr)

    def update(self) -> torch.Tensor:
        if len(self.mem) < self.hyp_p.batch_size:
            return torch.tensor([0])
        transitions = self.mem.sample(self.hyp_p.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.etc_p.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.hyp_p.batch_size, device=self.etc_p.device)
        next_state_values[non_final_mask] = self.agent.target_net(
            non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.hyp_p.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optim.zero_grad()
        loss.backward()
        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        return loss
