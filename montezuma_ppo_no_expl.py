import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import csv
import re 
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import cv2
from hashlib import sha256
from collections import OrderedDict
import matplotlib
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from gym import spaces
from stable_baselines3.common.callbacks import CheckpointCallback

# Create environment
env = gym.make('MontezumaRevenge', render_mode="rgb_array")
env1 = GrayScaleObservation(env, keep_dim=True)
env1 = DummyVecEnv([lambda: env1])
env1 = VecFrameStack(env1, 4, channels_order='last')

def convert_state(state):
    state = state.squeeze()
    height, width, num_frames = state.shape
    new_width = 8
    new_height = 11
    depth = 12  
    resized_frames = []
    for i in range(num_frames):
        resized_frame = cv2.resize(state[:, :, i], (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_frame = ((resized_frame / 255.0) * depth).astype(np.uint8)
        resized_frames.append(resized_frame)
    resized_state = np.stack(resized_frames, axis=-1)
    return resized_state.astype(np.uint8)

def make_reference(cell):
    cell_as_string = ''.join(cell.astype(int).astype(str).flatten())
    cell_as_bytes = cell_as_string.encode()
    cell_as_hash_bytes = sha256(cell_as_bytes)
    cell_as_hash_hex = cell_as_hash_bytes.hexdigest()
    cell_as_hash_int = int(cell_as_hash_hex, 16)
    cell_as_hash_string = str(cell_as_hash_int)
    return cell_as_hash_string

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'no_expl_ppo_montezuma_\d+_steps.zip', f)]
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: int(re.findall(r'(\d+)_steps', x)[0]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0])

class LogCallback(BaseCallback):
    def __init__(self, verbose=1, log_file='training_log.csv', log_file2='episode_info.csv'):
        super(LogCallback, self).__init__(verbose)
        self.visited_cells = set()
        self.visited_cells_per_episode = set()
        self.current_episode_reward = 0
        self.score = 0
        self.episode_count = 0
        self.iteration_count = 0
        self.log_file = log_file
        self.log_file2 = log_file2
        self.train_keys = ['train/entropy_loss', 'train/policy_gradient_loss', 
                           'train/value_loss', 'train/approx_kl', 'train/clip_fraction', 
                           'train/loss', 'train/explained_variance']
        self.metrics = []
        self._initialize_log_files()

    def _initialize_log_files(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Iteration', 'Episode', 'Cells', 'Score'] + self.train_keys)

        if not os.path.exists(self.log_file2):
            with open(self.log_file2, 'w', newline='') as ep_file:
                writer = csv.writer(ep_file)
                writer.writerow(['Episode', 'Cells per ep', 'Reward per ep', 'Score'])

    def save_state(self, save_path):
        state = {
            'visited_cells': self.visited_cells,
            'visited_cells_per_episode': self.visited_cells_per_episode,
            'current_episode_reward': self.current_episode_reward,
            'score': self.score,
            'episode_count': self.episode_count,
            'iteration_count': self.iteration_count
        }
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"LogCallback state saved to {save_path}")

    def load_state(self, load_path):
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        self.visited_cells = state['visited_cells']
        self.visited_cells_per_episode = state['visited_cells_per_episode']
        self.current_episode_reward = state['current_episode_reward']
        self.score = state['score']
        self.episode_count = state['episode_count']
        self.iteration_count = state['iteration_count']
        print(f"LogCallback state loaded from {load_path}")

    def _on_step(self) -> bool:
        obs = self.locals['new_obs']
        cell = convert_state(obs) 
        ref = make_reference(cell)
        self.visited_cells.add(ref)
        self.visited_cells_per_episode.add(ref)
        
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        self.score += reward

        if self.locals['dones'][0]:
            self.episode_count += 1
            with open(self.log_file2, 'a', newline='') as ep_file:
                writer = csv.writer(ep_file)
                writer.writerow([
                    self.episode_count,
                    len(self.visited_cells_per_episode),
                    self.current_episode_reward,
                    self.score
                ])
     
            self.current_episode_reward = 0
            self.visited_cells_per_episode.clear()
        return True

    def _on_rollout_end(self) -> None:
        metrics = {key: self.model.logger.name_to_value.get(key, 0.0) for key in self.train_keys}
        self.metrics.append(metrics)
        self.iteration_count += 1

        if self.verbose > 0:
            print(f"Iteration {self.iteration_count}: {metrics}")
            print(f"Episode {self.episode_count}, Cells Discovered: {len(self.visited_cells)}, Score: {self.score}")
        
        with open(self.log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.iteration_count,
                self.episode_count, 
                len(self.visited_cells), 
                self.score
            ] + [metrics.get(key, 0.0) for key in self.train_keys])

class CustomCheckpointCallback(CheckpointCallback):
    def __init__(self, start_step=0, log_callback=None, *args, **kwargs):
        super(CustomCheckpointCallback, self).__init__(*args, **kwargs)
        self.start_step = start_step
        self.log_callback = log_callback

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps + self.start_step
        if total_steps % self.save_freq == 0:
            save_path = os.path.join(self.save_path, f"{self.name_prefix}_{total_steps}_steps.zip")
            self.model.save(save_path)
            print(f"Saving model checkpoint to {save_path}")
            if self.log_callback is not None:
                log_state_path = os.path.join(self.save_path, 'log_callback_state.pkl')
                self.log_callback.save_state(log_state_path)
        return True

checkpoint_dir = './no_exploration_ppo_models/'
os.makedirs(checkpoint_dir, exist_ok=True)
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
state_file_path = os.path.join(checkpoint_dir, 'log_callback_state.pkl')

if latest_checkpoint:
    model1 = PPO.load(latest_checkpoint, env=env1)
    last_checkpoint_step = int(re.findall(r'(\d+)_steps', latest_checkpoint)[0])
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    log_callback = LogCallback()
    if os.path.exists(state_file_path):
        log_callback.load_state(state_file_path)
    else:
        print("No previous log callback state found. Starting fresh.")
else:
    model1 = PPO('CnnPolicy', env1, learning_rate=2.5e-4, gamma=0.99, verbose=1)
    last_checkpoint_step = 0
    log_callback = LogCallback()
    print("Starting new training")

total_timesteps = 10000000
remaining_timesteps = total_timesteps - last_checkpoint_step

checkpoint_callback = CustomCheckpointCallback(
    start_step=last_checkpoint_step,
    save_freq=2048,
    save_path=checkpoint_dir,
    name_prefix='no_expl_ppo_montezuma',
    log_callback=log_callback
)

model1.learn(total_timesteps=remaining_timesteps, callback=[log_callback, checkpoint_callback])
