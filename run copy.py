import torch
from replay_memory import ReplayMemory
from parameter import HyperParameters
from wrapped_cartpole import WrappedCartPole
from teacher import Teacher
from progress_plot import ProgressPlot
from agent import Agent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim_type = torch.optim.RMSprop
    
    # Aggregate
    hp = HyperParameters(
        image_w=60,
        image_h=60,
        output_size=2,
        device=device,
        target_update_interval=50,
        replay_memory_size=100000,
        num_episodes=2000,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=200,
        batch_size=128,
        lr=0.001,
        optimizer_type=optim_type,
        gamma=0.999,
        max_steps=200,
        save_interval=500,
        save_path="checkpoints/",
        plot_length = 5000
    )
    display = False
    initial_checkpoint = ""

    plotter = ProgressPlot(plot_length=hp.plot_length, cut=True)
    
    mem = ReplayMemory(hp.replay_memory_size)
    env = WrappedCartPole(hp)
    agent = Agent(hp, initial_checkpoint)
    teacher = Teacher(hp, agent, mem)
    
    # Records
    mean_duration = 100
    reward_record = []
    mean_record = []
    loss_record = []
    
    for episode_i in range(hp.num_episodes):
        
        state = env.reset()
        accumulated_reward = 0
        
        for j in range(hp.max_steps):
            action = agent.select_action(state)
            agent.step()
            next_state, reward, done, _, life = env.step(action.item())
                        
            reward_tensor = torch.tensor([reward], device=hp.device)
            mem.push(state, action, next_state, reward_tensor)
            state = next_state
            
            
            accumulated_reward += life         
            
            if display:
                env.render()
            
            if done:
                
                loss_tensor = teacher.update()
                loss_record.append(loss_tensor.item())   
                
                reward_record.append(accumulated_reward)
                lasts = reward_record[-mean_duration:]
                mean = sum(lasts)/len(lasts)
                mean_record.append(mean)                
                
                plotter.plot(reward_record, mean_record, loss_record)
                break
        
        if episode_i % hp.target_update_interval == 0:
            agent.update_target_net()

        if (episode_i + 1) % hp.save_interval == 0:
            agent.save(hp.save_path + f"model_ep{episode_i}.pt")
    
    