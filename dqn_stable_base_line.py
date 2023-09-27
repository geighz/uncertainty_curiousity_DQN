#from stable_baselines3 import DQN
from curiousity_DQN import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import multiprocessing
import time
env_id = 'Pong-ram-v4'
#env_id = 'CartPole-v1'
cpu_num = multiprocessing.cpu_count()
vec_env = make_vec_env(env_id,n_envs = cpu_num)

model = DQN('MlpPolicy', vec_env, verbose=0)

n_timesteps = 200_000_000

# Multiprocessed RL Training
start_time = time.time()
model.learn(n_timesteps)
total_time_multi = time.time() - start_time



print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))

# # Single Process RL Training
# single_process_model = DQN('MlpPolicy', env_id, verbose=0)

# start_time = time.time()
# single_process_model.learn(n_timesteps)
# total_time_single = time.time() - start_time

# print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time_single, n_timesteps / total_time_single))

# print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))