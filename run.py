import warnings
warnings.filterwarnings("ignore")

import torch
from replay_memory import ReplayMemory
from parameter import HyperParameters, EnvParameters, ETCParameters
from teacher import Teacher
from progress_plot import ProgressPlot
from agent import Agent
from utils import get_env_list
import gym
import torchvision.transforms as T
from PIL import Image
from datetime import datetime
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim_type = torch.optim.Adam
    
    hyp_p = HyperParameters(
        target_update_interval=50,
        replay_memory_size=1000000,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay=100000,
        batch_size=32,
        lr=0.0001,
        optimizer_type=optim_type,
        gamma=0.99,
        num_episodes=3000,
        policy_update_step_interval=500,
        image_size=None,
        max_steps=5000,
        frame_skips=3
    )
    
    etc_p = ETCParameters(
        device=device,
        save_interval=500,
        save_path="checkpoints/",
        plot_length=5000,
    )
    
    if hyp_p.image_size:
        transform = T.Compose([
            T.ToPILImage(), 
            T.Resize((hyp_p.image_size, hyp_p.image_size), interpolation=Image.CUBIC), 
            T.Grayscale(),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(), 
            T.Grayscale(),
            T.ToTensor(),
        ])

    
    display = False
    initial_checkpoint = ""
    #env_names = get_env_list()
    env_names = ['ALE/Breakout-v5', 
                 'ALE/Asteroids-v5', 
                 'ALE/DonkeyKong-v5', 
                 'ALE/SpaceInvaders-v5', 
                 'ALE/Pong-v5']
    
    for env_name in env_names:
        print(f"Running on {env_name}")
        env_start = datetime.now()
        
        env = gym.make(env_name)
        env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        w, h, c = state.shape
        
        env_p = EnvParameters(
            image_w=w,
            image_h=h,
            output_size=env.action_space.n
        )

        mem = ReplayMemory(hyp_p.replay_memory_size)
        
        agent = Agent(hyp_p, env_p, etc_p, initial_checkpoint, use_multi_gpu=True)
        teacher = Teacher(hyp_p, etc_p, agent, mem)
        plotter = ProgressPlot(plot_length=etc_p.plot_length, cut=True)
        
        # Records
        mean_duration = 100
        reward_record = []
        mean_record = []
        loss_record = []
        
        for episode_i in range(hyp_p.num_episodes):
            
            env.reset()
            action = env.action_space.sample()
            image, _, _, _ = env.step(action)
            state = transform(image).unsqueeze(0)
            accumulated_reward = 0
                        
            for steps in range(hyp_p.max_steps):
                if steps % hyp_p.frame_skips == 0:
                    action = agent.select_action(state)
                agent.step()

                next_image, reward, done, info = env.step(action.item())
                # Clip reward
                if reward > 0:
                    reward = 1
                if reward < 0:
                    reward = -1
                next_state = transform(next_image).unsqueeze(0)
                
                reward_tensor = torch.tensor([reward], device=etc_p.device)
                mem.push(state, action, next_state, reward_tensor)
                state = next_state
                
                accumulated_reward += reward
                
                if display:
                    env.render()
                
                if (steps + 1) % hyp_p.policy_update_step_interval == 0:
                    loss_tensor = teacher.update()
                    loss_record.append(loss_tensor.item())   

                if done:
                    loss_tensor = teacher.update()
                    loss_record.append(loss_tensor.item())   
                    
                    reward_record.append(accumulated_reward)
                    lasts = reward_record[-mean_duration:]
                    mean = sum(lasts)/len(lasts)
                    mean_record.append(mean)                
                    
                    plotter.save(
                        os.path.join(etc_p.save_path, env_name),
                        f"result.png", 
                        reward_record, 
                        mean_record, 
                        loss_record
                    )
                    break
            
            if episode_i % hyp_p.target_update_interval == 0:
                agent.update_target_net()

            if (episode_i + 1) % etc_p.save_interval == 0:
                checkpoint_name = f"{env_name}_{mean:.4f}.pt"
                agent.save(etc_p.save_path + checkpoint_name)
        
        env_end = datetime.now()
        elapsed = (env_end - env_start).total_seconds()
        max_reward = max(reward_record)
        mean_reward = max(mean_record)
        print(f"{env_name} - Elasped: {elapsed:.2f}[s] | Max R: {max_reward:.2f} | Max Mean R: {mean_reward:.2f}")
        plotter.save(
            f"{etc_p.save_path}{env_name}_result.png", 
            reward_record, 
            mean_record, 
            loss_record
        )
        plotter.close()
