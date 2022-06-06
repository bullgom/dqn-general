import gym



def get_env_list() -> list[str]:
    tree = gym.envs.registry.all()._mapping.tree["ALE"]

    env_names = [f"ALE/{name}-v5" for name, value in tree.items() if "ram" not in name]
    
    return env_names

