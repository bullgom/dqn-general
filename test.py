import gym

tree = gym.envs.registry.all()._mapping.tree["ALE"]

env_names = [f"ALE/{name}-v5" for name, value in tree.items() if "ram" not in name]

print(env_names)

sizes = []

for name in env_names:
    
    env = gym.make(name)
    env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    sizes.append(state.shape)
    env.close()

print(sizes)
