#!/usr/bin/env python3

###
# Small test script for PyTorch and its Ignite library
###

import datetime
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler

import typing as ty
from dataclasses import dataclass

HIDDEN_WIDTH = 10
HIDDEN_DEPTH = 1
BATCH_SIZE = 32
PERCENTILE = 80

LEARN_RATE = 0.05

log = gym.logger
log.set_level(gym.logger.INFO)


class TestNW(nn.Module):
    def __init__(self, obs: int, hidden_depth: int, hidden_width: int, n_actions: int):
        super(TestNW, self).__init__()
        layers = []
        layers.append(nn.Linear(obs, hidden_width))
        for i in range(1, hidden_depth):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers, nn.Linear(hidden_width, n_actions))

    def forward(self, x: torch.Tensor):
        return self.network(x)


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: ty.List[EpisodeStep]


def generate_batches(env: gym.Env, model: nn.Module):
    model.eval()

    batches = []
    rewards = 0.0
    steps = []

    obs, _ = env.reset()
    softmax = nn.Softmax(dim=1).cuda()

    is_done = False
    is_trunc = False

    while True:
        obs_ft = torch.FloatTensor(obs).cuda()

        action_probs = softmax(model(obs_ft.unsqueeze(0)))
        action_probs = action_probs.cpu().data.numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)

        obs_n, reward, is_done, is_trunc, _ = env.step(action=action)

        rewards += float(reward)

        steps.append(EpisodeStep(obs, action))

        if is_done or is_trunc:
            batches.append(Episode(rewards, steps))

            rewards = 0.0
            steps = []
            obs_n, _ = env.reset()
            if BATCH_SIZE == len(batches):
                yield batches
                batches = []

        obs = obs_n


def filter_batches(
    batch: ty.List[Episode],
) -> ty.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    episode_rewards = list(map(lambda ep: ep.reward, batch))

    reward_cutoff = float(np.percentile(episode_rewards, PERCENTILE))
    reward_mean = float(np.mean(episode_rewards))

    best_observations = []
    best_actions = []

    for episode in batch:
        if episode.reward < reward_cutoff:
            continue

        best_observations.extend(map(lambda step: step.observation, episode.steps))
        best_actions.extend(map(lambda step: step.action, episode.steps))

    obs_for_training = torch.FloatTensor(np.vstack(best_observations))
    actions_for_training = torch.LongTensor(best_actions)

    return obs_for_training, actions_for_training, reward_cutoff, reward_mean


class Trainer:
    def __init__(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        loss_fn: nn.Module,
    ):
        self.optimizer = optimizer
        self.model = model
        self.loss = loss_fn

    def __call__(self, engine: Engine, batch: ty.Any):
        self.train_episode(engine, batch)

    def train_episode(self, engine: Engine, batch: ty.Any):
        obs, acts, reward_cutoff, reward_mean = filter_batches(batch)

        self.model.train()

        self.optimizer.zero_grad()
        action_v = self.model(obs.cuda())
        loss = self.loss(action_v, acts.cuda())
        loss.backward()
        self.optimizer.step()

        self.model.eval()

        engine.state.metrics = {
            "iteration": engine.state.iteration,
            "loss": loss.item(),
            "r_cutoff": reward_cutoff,
            "r_mean": reward_mean,
        }

        # Cartpole rewards are truncated at 500
        # Considering it solved with mean reward of 475 seems reasonable
        if reward_mean > 475:
            print(f"Solved :)")
            engine.terminate_epoch()


def eval_network(model: nn.Module, env: RecordVideo):
    model.eval()
    softmax = nn.Softmax(dim=1).cuda()

    obs, _ = env.reset()

    is_done = False
    is_trunc = False

    while not (is_done or is_trunc):
        obs_ft = torch.FloatTensor(obs).cuda()

        action_probs = softmax(model(obs_ft.unsqueeze(0)))
        action_probs = action_probs.cpu().data.numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)

        obs, _, is_done, is_trunc, _ = env.step(action=action)

    env.close_video_recorder()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]

    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    result_directory = (
        "./_results/cartpole/01_crossentropy/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    env_validation = RecordVideo(
        gym.make("CartPole-v1", render_mode="rgb_array"),
        result_directory + "/videos",
        name_prefix="01_crossentropy",
        disable_logger=True,
        episode_trigger=lambda i: True,
    )

    network = TestNW(
        obs=obs_size,
        hidden_depth=HIDDEN_DEPTH,
        hidden_width=HIDDEN_WIDTH,
        n_actions=n_actions,
    )
    print(network)
    network = network.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=network.parameters(), lr=LEARN_RATE)

    trainer = Trainer(optimizer, network, loss_function)

    timer = Timer()

    with TensorboardLogger(log_dir=result_directory + "/metrics") as logger:

        engine = Engine(trainer)

        timer.attach(engine)

        metrics_handler = OutputHandler(
            tag="Training", metric_names=["loss", "r_cutoff", "r_mean"]
        )

        logger.attach(
            engine=engine,
            log_handler=metrics_handler,
            event_name=Events.ITERATION_COMPLETED,
        )

        @engine.on(Events.ITERATION_COMPLETED)
        def log_on_iteration_completed(e: Engine):
            log.info(
                f"{e.state.metrics['iteration']:2d} in {timer.value():.2f}s: "
                f"loss= {e.state.metrics['iteration']:.5f}, "
                f"r_cutoff= {e.state.metrics['r_cutoff']:.1f}, "
                f"r_mean= {e.state.metrics['r_mean']:.1f}"
            )

            if not e.state.metrics["iteration"] % 5:
                eval_network(network, env_validation)

            timer.reset()

        @engine.on(Events.EPOCH_COMPLETED)
        def log_on_epoch_completed(e: Engine):
            if e.state.metrics["iteration"] % 5:
                eval_network(network, env_validation)

        engine.run(data=generate_batches(env, network))
