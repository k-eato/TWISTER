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

# Solve dm_control bug
import os
os.environ["MUJOCO_GL"] = "egl"

# PyTorch
import torch

# Functions
import functions

# Other
import os
import argparse
import importlib
import warnings
import datetime

import nnet
from nnet import embodied
from nnet.embodied.envs import from_gym
import ruamel.yaml as yaml

from nnet import car_dreamer

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

# Disable Warnings
warnings.filterwarnings("ignore")

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

def main(args=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "configs/config.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(args)
    for name in parsed.task:
        env, env_config = car_dreamer.create_task(name, args)
        config = config.update(env_config)
    config = embodied.Flags(config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            embodied.logger.WandBOutput(logdir.name, config)
        ],
    )

    dreamerv3_config = config.dreamerv3
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    # agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamerv3_config)
    # replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    # args = embodied.Config(
    #     **dreamerv3_config.run,
    #     logdir=dreamerv3_config.logdir,
    #     batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
    #     actor_dist_disc=dreamerv3_config.actor_dist_disc,
    # )
    # embodied.run.train(agent, env, replay, logger, args)

    args = config

    # Print Mode
    print("Mode: {}".format(args.dreamerv3.twister.mode))

    # Load Config
    env_name = "carla"
    callback_path = "callbacks/{}".format(env_name)

    # Model
    model = nnet.models.TWISTER(env_name=env_name, env=env, cnn_keys=args.dreamerv3.encoder["cnn_keys"], override_config={})
    model.compile()

    # Training
    precision = model.config.precision
    grad_init_scale = model.config.grad_init_scale
    epochs = model.config.epochs
    epoch_length = model.config.epoch_length


    # Replay Buffer
    training_dataset = nnet.datasets.ReplayBuffer(
        batch_size=model.config.batch_size,
        root=callback_path,
        buffer_capacity=model.config.buffer_capacity,
        epoch_length=epoch_length,
        sample_length=model.config.L
    )
    model.set_replay_buffer(training_dataset)

    # Evaluation Dataset
    evaluation_dataset = nnet.datasets.VoidDataset(num_steps=model.config.eval_episodes)

    # Load Model
    model = functions.load_model(args, model, callback_path)

    # Load Dataset
    dataset_train, dataset_eval = functions.load_datasets(training_dataset, evaluation_dataset)

    ###############################################################################
    # Modes
    ###############################################################################

    # Training
    if args.dreamerv3.twister.mode == "training":
        model.fit(
            dataset_train=dataset_train, 
            epochs=epochs, 
            dataset_eval=dataset_eval, 
            initial_epoch=int(args.dreamerv3.twister.checkpoint.split("_")[2]) if args.dreamerv3.twister.checkpoint != "None" else 0, 
            callback_path=callback_path,
            precision=precision,
            accumulated_steps=1,
            eval_period_step=args.dreamerv3.twister.eval_period_step,
            eval_period_epoch=args.dreamerv3.twister.eval_period_epoch,
            saving_period_epoch=args.dreamerv3.twister.saving_period_epoch,
            log_figure_period_step=args.dreamerv3.twister.log_figure_period_step,
            log_figure_period_epoch=args.dreamerv3.twister.log_figure_period_epoch,
            step_log_period=args.dreamerv3.twister.step_log_period,
            grad_init_scale=grad_init_scale,
            detect_anomaly=args.dreamerv3.twister.detect_anomaly,
            recompute_metrics=False,
            wandb_logging=args.dreamerv3.twister.wandb,
            verbose_progress_bar=args.dreamerv3.twister.verbose_progress_bar,
            keep_last_k=args.dreamerv3.twister.keep_last_k
        )

    # Evaluation
    elif args.dreamerv3.twister.mode == "evaluation":

        model._evaluate(
            dataset_eval, 
            writer=None,
            recompute_metrics=False,
            verbose_progress_bar=args.dreamerv3.twister.verbose_progress_bar,
        )

    # Pass
    elif args.dreamerv3.twister.mode == "pass":
        pass

if __name__ == "__main__":
    main()
