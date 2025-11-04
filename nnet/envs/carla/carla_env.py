# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torchvision

# NeuralNets
from nnet.structs import AttrDict
from nnet.embodied.envs import from_gym
from nnet import embodied
# Gym
import gym.wrappers

from ruamel.yaml import YAML

# Other
import os
import datetime


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


class CarlaEnv:
    def obs_space(self):

        if self.grayscale_obs:
            return (["image", (1 * self.history_frames, self.img_size[0], self.img_size[1]), torch.uint8],)
        else:
            return (["image", (3 * self.history_frames, self.img_size[0], self.img_size[1]), torch.uint8],)

    def __init__(
            self,
            env,
            img_size=(64, 64),
            cnn_keys = None,
            action_repeat=4, 
            history_frames=1, 
            seed=None, 
            repeat_action_probability=0.0, 
            episode_saving_path=None, 
            noop_max=30, 
            terminal_on_life_loss=False, 
            grayscale_obs=False, 
            full_action_space=False
        ):

        # Params
        self.img_size = img_size
        self.grayscale_obs = grayscale_obs
        self.terminal_on_life_loss = terminal_on_life_loss
        self.env = env
        self.cnn_keys = cnn_keys.split("|")
        
        # Params
        self.episode_saving_path = episode_saving_path
        if self.episode_saving_path is not None:
            if not os.path.isdir(self.episode_saving_path):
                os.makedirs(self.episode_saving_path, exist_ok=True)
        self.grayscale_obs = grayscale_obs
        self.action_repeat = action_repeat
        self.history_frames = history_frames

        # Set Seed
        self.seed(seed)

        # Default Action Space
        self.num_actions = self.env.act_space['action'].shape[0]

        # FPS
        self.fps = 60.0 / self.action_repeat

    def seed(self, seed):
        if seed:
            self.env.seed(seed)

    def sample(self):

        return torch.nn.functional.one_hot(torch.randint(low=0, high=self.num_actions, size=()), num_classes=self.num_actions).type(torch.float32)

    def preprocess(self, state, reward, done):
        # Convert state dict to array

        # To tensor
        state = torch.cat([state[k] for k in self.cnn_keys], -1)
        state = torch.squeeze(state, dim=0)
        print(state.size())

        # (C, H, W)
        if self.grayscale_obs:
            state = state.unsqueeze(dim=0)
        else:
            state = state.permute(2, 0, 1)

        # Reward
        reward = torch.tensor(reward, dtype=torch.float32)

        # Done
        done = torch.tensor(done, dtype=torch.float32)

        # Is_last
        if self.terminal_on_life_loss:
            is_last = torch.tensor(self.env.env.ale.game_over(), dtype=torch.float32)
        else:
            is_last = done

        return state, reward, done, is_last

    def reset(self):

        # Reset
        state, _, _, _ = self.preprocess(self.env.reset(), 0, 0)
        
        # Episode videos
        if self.episode_saving_path is not None:
            self.episode_video = []
            self.episode_video_pre = []

        # Repeat history frames along channels
        if self.history_frames > 1:
            self.history = state.repeat(self.history_frames, 1, 1)

        # No history frames
        else:
            self.history = state

        # Episode Score
        self.episode_score = 0.0

        # Reward
        reward = torch.tensor(0.0, dtype=torch.float32)

        # Done
        done = torch.tensor(False, dtype=torch.float32)

        # Is_last
        is_last = torch.tensor(False, dtype=torch.float32)

        # Is First
        is_first = torch.tensor(True, dtype=torch.float32)

        return AttrDict(state=self.history, reward=reward, done=done, is_first=is_first, is_last=is_last)

    def step(self, action):

        # Env Step
        state, reward, done, infos = self.env.step(action.item())

        # Update Episode Score
        self.episode_score += reward

        # Add to video
        if self.episode_saving_path is not None:
            self.episode_video.append(torch.tensor(self.env.env._get_image()))
            self.episode_video_pre.append(torch.tensor(torch.squeeze(state['birdeye_wpt'], dim=0)))

        # Save Episode
        if done and self.episode_saving_path is not None:

            # Stack videos
            self.episode_video = torch.stack(self.episode_video, dim=0)
            self.episode_video_pre = torch.stack(self.episode_video_pre, dim=0)

            # Datetime
            date_time_score = str(datetime.datetime.now()).replace(" ", "_") + "_" + str(self.episode_score)

            # Save Videos
            torchvision.io.write_video(filename=os.path.join(self.episode_saving_path, "{}.mp4".format(date_time_score)), video_array=self.episode_video, fps=self.fps, video_codec="libx264")
            torchvision.io.write_video(filename=os.path.join(self.episode_saving_path, "{}_pre.mp4".format(date_time_score)), video_array=self.episode_video_pre.unsqueeze(dim=-1).repeat(1, 1, 1, 3) if self.grayscale_obs else self.episode_video_pre, fps=self.fps, video_codec="libx264")

        # Preprocessing
        state, reward, done, is_last = self.preprocess(state, reward, done)

        # Is First
        is_first = torch.tensor(False, dtype=torch.float32)

        # Concat history frames along channels
        if self.history_frames > 1:
            if self.grayscale_obs:
                self.history = torch.cat([self.history[1:], state], dim=0)
            else:
                self.history = torch.cat([self.history[3:], state], dim=0)

        # No history frames
        else:
            self.history = state

        return AttrDict(state=self.history, reward=reward, done=done, is_first=is_first, is_last=is_last)