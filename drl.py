import os
from env_perlin import Hexapod as env
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.utils import get_device


# get_device("cuda")
DRL = 1  # 1：SAC   2: PPO   3:TD3
train = False
# train = False
policy_show = False
log_dir = "./result/perlin"

os.makedirs(log_dir, exist_ok=True)
if DRL == 1:
    model_name = "SAC"
elif DRL == 2:
    model_name = "PPO"
elif DRL == 3:
    model_name = "TD3"
else:
    print("DRL error")
    exit(0)
file_path = os.path.join(log_dir, model_name)
env = env()
if train:
    if DRL == 1:
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                    learning_rate=0.0003, buffer_size=1_000_000, batch_size=256, gamma=0.99,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])))
    elif DRL == 2:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                    learning_rate=0.0002, batch_size=128, gae_lambda=0.95, gamma=0.99,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    elif DRL == 3:
        model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                    learning_rate=0.001, buffer_size=1_000_000, batch_size=128, gamma=0.99,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 256], qf=[512, 256])))
    else:
        print("DRL error")
        exit(0)
    if policy_show:
        policy = model.policy
        print(policy)
        exit(0)
    model.learn(total_timesteps=int(6e6), progress_bar=True)
    model.save(file_path)
    del model
if DRL == 1:
    model = SAC.load(file_path, env=env)
elif DRL == 2:
    model = PPO.load(file_path, env=env)
elif DRL == 3:
    model = TD3.load(file_path, env=env)
else:
    print("DRL error")
    exit(0)
# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
r = 0
for i in range(1):
    for j in range(399):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        r += rewards
        env.render()
    print("reward", r)
    r = 0
