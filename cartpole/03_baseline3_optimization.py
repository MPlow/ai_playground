#!/usr/bin/env python3

###
# Small test script for a optimization using Optuna of the previous CartPole script
###

import os
import datetime

import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch.nn as nn

import typing as ty


VERBOSE_LEVEL = 0

N_TRAINING_ENVS = 8
N_EVAL_ENVS = 1

VIDEO_LENGTH_STEPS = 300

N_TRIALS = 1000
N_STARTUP_TRIALS = 5
N_WARMUP_STEPS = 3


class TrialCallback(EvalCallback):
    def __init__(
        self,
        trial: optuna.Trial,
        eval_env: ty.Union[gym.Env, VecEnv],
        best_model_save_path: ty.Optional[str],
        log_path: ty.Optional[str],
        n_eval_episodes: int,
        eval_freq: int,
        deterministic: bool,
    ):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
        )

        self.trial = trial
        self.prune_trial = False
        self.eval_idx = 0

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()

            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.prune_trial = True
                return False

        return True


def sample_parameters(trial: optuna.Trial) -> dict[str, ty.Any]:
    net_arch = []
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layer_width = trial.suggest_int("layer_width", 1, 129)
    for i in range(n_layers):
        net_arch.append(layer_width)

    activation_dict = {
        "relu": nn.ReLU,
        "prelu": nn.PReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
    }

    activation_fn = trial.suggest_categorical(
        "activation_fn", list(activation_dict.keys())
    )
    activation_fn = activation_dict[activation_fn]

    gamma = trial.suggest_float("gamma", 0.97, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 3e-6, 3e-1, log=True)

    n_steps_pow = trial.suggest_int("n_steps_pow", 4, 10)
    n_steps = 2**n_steps_pow

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "policy_kwargs": {
            "activation_fn": activation_fn,
            "net_arch": net_arch,
        },
    }


if __name__ == "__main__":
    env_id = "CartPole-v1"

    study_directory = "_results//cartpole//03_baseline3_optimization"

    result_directory = (
        "./" + study_directory + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    def objective(trial: optuna.Trial) -> float:
        training_envs = make_vec_env(env_id=env_id, n_envs=N_TRAINING_ENVS, seed=0)
        eval_envs = make_vec_env(env_id=env_id, n_envs=N_EVAL_ENVS, seed=0)

        logger = configure(
            result_directory + "/metrics",
            ["tensorboard"],
        )

        eval_callback = TrialCallback(
            trial=trial,
            eval_env=eval_envs,
            best_model_save_path=result_directory + "/models",
            log_path=result_directory + "/eval_metrics",
            n_eval_episodes=5,
            eval_freq=max(1_000 // N_EVAL_ENVS, 1),
            deterministic=True,
        )

        model = A2C(
            "MlpPolicy",
            training_envs,
            verbose=VERBOSE_LEVEL,
            **sample_parameters(trial),
        )
        model.set_logger(logger)
        model.learn(total_timesteps=100_000, callback=eval_callback)

        if eval_callback.prune_trial:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_WARMUP_STEPS
    )

    os.makedirs(study_directory, exist_ok=True)

    study = optuna.create_study(
        storage=f"sqlite:///{study_directory}/db.sqlite3",
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("Aborted study")
        pass

    trial = study.best_trial

    print(f"Finished trials: {len(study.trials)}")
    print(f"Best trial: {trial.value}")
    print(f" Parameter:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
