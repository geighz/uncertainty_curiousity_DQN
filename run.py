#from stable_baselines3 import DQN
from curiousity_DQN import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import multiprocessing
import time
import os
import argparse
import datetime
import logging
import gymnasium

def run():
    # env_id = 'Pong-ram-v4'
    # env_id = 'CartPole-v1'
    models_dir = "models/DQN"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--buffer_path',type=str)
    parser.add_argument('--buffer_dir', type=str)
    parser.add_argument('--num_timesteps_per_save', type=int, default=50_000_000)
    parser.add_argument('--env_id', type=str, default='ALE/Breakout-ram-v5')
    parser.add_argument('--exploration_mode',type=str, default='Thompson')
    parser.add_argument('--exploitation_mode', type=str, default='Normal')

    args = parser.parse_args()

    if args.buffer_dir == None:
        print('Error! Need to set home directory for saving buffer data (--buffer_dir)!')
        return 1
    
  
    env_id = args.env_id
    cpu_num = multiprocessing.cpu_count()
    vec_env = make_vec_env(env_id,n_envs = cpu_num)#,wrapper_class=gymnasium.wrappers.Monitor,wrapper_kwargs={"/path/to/folder/", force=True})
    eval_env = make_vec_env(env_id, n_envs=cpu_num)
    timesteps_per_save = args.num_timesteps_per_save
    TIMESTEPS = timesteps_per_save
    iters = 0
    if args.model_path:
        #Load model  
        path_variables = args.model_path.split('/')
        path_variables = path_variables[-1].split('-')[-1].split('.')[0]
        print(path_variables)

        iters = int(int(path_variables)/timesteps_per_save)
        model = DQN.load(args.model_path,vec_env,verbose=0)
        if args.buffer_path:
            model.load_replay_buffer(args.buffer_path)
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")
        else:
            print('Warning! Replay buffer not initialized.')
        #model.exploration_initial_eps = 1
       
    else:
        model = DQN('MlpPolicy', vec_env, verbose=0)
        model.q_net.exploration_mode = args.exploration_mode
        model.q_net.exploitation_mode = args.exploitation_mode
    
    print('Time Started:',datetime.datetime.now())
    # Multiprocessed RL Training
    
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, progress_bar=True,tb_log_name=f"{args.env_id}/{args.exploration_mode}-{args.exploitation_mode}-{timesteps_per_save*iters}forever")
        model.save(f"{models_dir}/{args.env_id}/{args.exploration_mode}-{args.exploitation_mode}-{timesteps_per_save*iters}")
        model.save_replay_buffer(f"{args.buffer_dir}/{args.env_id}/{args.exploration_mode}-{args.exploitation_mode}-{timesteps_per_save*iters}")
        
        # Evaluate the loaded policy
        mean_reward, std_reward = evaluate_policy(model.policy, eval_env, n_eval_episodes=1, deterministic=True)
        print(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward}")
        print("Last model:",args.exploration_mode,args.exploitation_mode,env_id,"saved at total run time:",datetime.datetime.now(),"Iter:",iters)

if __name__ == '__main__':
    run()

#print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))

# # Single Process RL Training
# single_process_model = DQN('MlpPolicy', env_id, verbose=0)

# start_time = time.time()
# single_process_model.learn(n_timesteps)
# total_time_single = time.time() - start_time

# print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time_single, n_timesteps / total_time_single))

# print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))