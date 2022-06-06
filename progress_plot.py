import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ProgressPlot:
    
    def __init__(self, plot_length: int, cut : bool = False):
        self.plot_length = plot_length
        self.cut = cut
        fig, ax = plt.subplots(2)
        self.figure = fig
        self.rpe_axes = ax[0]
        self.loss_axes = ax[1]
        self.rpe_axes.set_xlabel("Episodes")
        self.rpe_axes.set_ylabel("Rewards")
        
        self.loss_axes.set_xlabel("Steps")
        self.loss_axes.set_ylabel("Loss")
        
        plt.tight_layout()
        plt.ion()
    
    def plot(self, rewards: list[float], mean_values: list[float], losses: list[float]) -> None:
        """
        Plots 
        1.
            (1) Reward per episode
            (2) Mean X evalues
        2.
            (1) Loss per step
        """
        if self.cut:
            rewards = rewards[-self.plot_length:]
            mean_values = mean_values[-self.plot_length:]
            losses = losses[-self.plot_length:]
        self.rpe_axes.clear()
        self.rpe_axes.set_title("Reward Per Episode")
        self.rpe_axes.plot(rewards, color="black", linewidth=1)
        self.rpe_axes.plot(mean_values, color="red",
                           linestyle="--", linewidth=1.5)
        self.rpe_axes.grid()        
        
        self.loss_axes.clear()
        self.loss_axes.set_title("Loss Per Episode")
        self.loss_axes.plot(losses, color="black",linewidth=1)
        self.loss_axes.grid()
        plt.pause(0.05)
        
    
    def close(self) -> None:
        plt.close()
    
    def save(
        self, 
        path: str,
        name: str,
        rewards: list[float], 
        mean_values: list[float], 
        losses: list[float]
    ) -> None:
        os.makedirs(path, exist_ok=True)
        self.rpe_axes.clear()
        self.rpe_axes.set_title("Reward Per Episode")
        self.rpe_axes.plot(rewards, color="black", linewidth=1)
        self.rpe_axes.plot(mean_values, color="red",
                           linestyle="--", linewidth=1.5)
        self.rpe_axes.grid()        
        
        self.loss_axes.clear()
        self.loss_axes.set_title("Loss Per Episode")
        self.loss_axes.plot(losses, color="black",linewidth=1)
        self.loss_axes.grid()
        x = os.path.join(path, name)
        if os.path.isfile(x):
            os.remove(x)
        plt.savefig(x)
