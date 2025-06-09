#!/usr/bin/env python3

###
# Small test script for a Baseline3 implementation of CartPole
###

import datetime

import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


VERBOSE = False

N_TRAINING_ENVS = 8
N_EVAL_ENVS = 2

VIDEO_LENGTH_STEPS = 300

if __name__ == "__main__":
    env_id = "CartPole-v1"

    training_envs = make_vec_env(env_id=env_id, n_envs=N_TRAINING_ENVS, seed=0)

    eval_envs = make_vec_env(env_id=env_id, n_envs=N_EVAL_ENVS, seed=0)

    result_directory = (
        "./_results/cartpole/02_baseline3/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    logger = configure(
        result_directory + "/metrics",
        ["stdout", "tensorboard"],
    )

    eval_callback = EvalCallback(
        eval_env=eval_envs,
        best_model_save_path=result_directory + "/models",
        log_path=result_directory + "/eval_metrics",
        n_eval_episodes=2,
        eval_freq=max(1_000 // N_TRAINING_ENVS, 1),
        deterministic=True,
    )

    model = A2C("MlpPolicy", training_envs, verbose=int(VERBOSE))
    model.set_logger(logger)
    model.learn(total_timesteps=100_000, callback=eval_callback)

    # Create video of finished model
    finished_model = A2C.load(result_directory + "/models/best_model.zip")

    vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

    _ = vec_env.reset()
    vec_env = VecVideoRecorder(
        venv=vec_env,
        video_folder=result_directory + "/videos",
        record_video_trigger=lambda x: x == 0,
        video_length=VIDEO_LENGTH_STEPS,
        name_prefix="finished_model",
    )

    obs = vec_env.reset()
    for _ in range(VIDEO_LENGTH_STEPS):
        action, _ = finished_model.predict(obs, deterministic=True)  # type: ignore
        obs, _, _, _ = vec_env.step(action)
    vec_env.close()
