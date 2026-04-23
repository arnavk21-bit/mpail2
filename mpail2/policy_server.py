"""
policy_server.py
Exposes MPAIL2Runner as an HTTP API so LeRobot (in a separate conda env)
can call it.

Run with:
    conda activate mpail2
    python policy_server.py
"""

import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── same imports as test_load.py ──────────────────────────────────────────────
import gymnasium as gym
from mpail2.runner import MPAIL2Runner
from mpail2.configs.cfgs import MPAIL2RunnerCfg
from mpail2.configs.defs import (
    MLPCoderConfig, PlannerConfig, PolicySamplingConfig, LearnerConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  –  edit these to match your task
# ─────────────────────────────────────────────────────────────────────────────
DEMO_PATH  = "envs/gym_mujoco/demos/Ant-v5/ppo_Ant-v5_20260202_152512/expert50ep.pt"
GYM_ENV_ID = "Ant-v5"
OBS_KEY    = "proprioception"
DEVICE     = "cpu"          # change to "cuda" on GPU server
LATENT_DIM = 512
NUM_ROLLOUTS = 512
NUM_ELITES   = 64
OPT_ITERS    = 5
MAX_EP_LEN   = 1000

# ─────────────────────────────────────────────────────────────────────────────
# BUILD RUNNER  (done once at startup)
# ─────────────────────────────────────────────────────────────────────────────

print("Building MPAIL2 runner...")

_env_raw    = gym.make(GYM_ENV_ID)
OBS_DIM     = _env_raw.observation_space.shape[0]
ACTION_DIM  = _env_raw.action_space.shape[0]

# --- env wrapper (same as test_load.py) ---
import numpy as _np

class GymDictWrapper(gym.Wrapper):
    def __init__(self, env, obs_key, max_episode_length, device="cpu"):
        super().__init__(env)
        self._obs_key = obs_key
        self.num_envs = 1
        self.max_episode_length = max_episode_length
        self.device = device
        _obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Dict({
            obs_key: gym.spaces.Box(
                low=-_np.inf, high=_np.inf,
                shape=(1, _obs_dim), dtype=_np.float32)
        })
    @property
    def unwrapped(self):
        return self
    def _wrap(self, obs):
        return {self._obs_key: torch.tensor(obs, dtype=torch.float32).unsqueeze(0)}
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap(obs), info
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).detach().cpu().numpy()
        obs, r, term, trunc, info = self.env.step(action)
        return (self._wrap(obs),
                torch.tensor([float(r)], dtype=torch.float32),
                torch.tensor([bool(term)], dtype=torch.bool),
                torch.tensor([bool(trunc)], dtype=torch.bool),
                info)

env = GymDictWrapper(_env_raw, OBS_KEY, MAX_EP_LEN, DEVICE)

# --- demos ---
raw = torch.load(DEMO_PATH, map_location="cpu", weights_only=False)
demo_tensor = raw if isinstance(raw, torch.Tensor) else (
    raw.get("data") or next(v for v in raw.values() if isinstance(v, torch.Tensor))
)
if demo_tensor.dim() == 2:
    demo_tensor = demo_tensor.unsqueeze(1).repeat(1, 2, 1)
demonstrations = {OBS_KEY: demo_tensor.float()}

# --- config ---
encoder_cfg  = MLPCoderConfig(obs_key=OBS_KEY, input_dim=OBS_DIM, output_dim=LATENT_DIM)
sampling_cfg = PolicySamplingConfig(num_rollouts=NUM_ROLLOUTS)
planner_cfg  = PlannerConfig(
    encoder_cfg=encoder_cfg, action_dim=ACTION_DIM,
    latent_dim=LATENT_DIM, sampling_cfg=sampling_cfg,
    opt_iters=OPT_ITERS, num_elites=NUM_ELITES,
)
learner_cfg = LearnerConfig(
    planner_cfg=planner_cfg,
    replay_size=100_000, replay_batch_size=256, use_terminations=False,
)
log_cfg = MPAIL2RunnerCfg.LogCfg(
    log_dir=None, checkpoint_every=999_999, no_wandb=True, video_interval=999_999,
)
runner_cfg = MPAIL2RunnerCfg(
    learner_cfg=learner_cfg, log_cfg=log_cfg,
    num_learning_iterations=100, logger=None, vis_rollouts=False,
)

runner = MPAIL2Runner(
    demonstrations=demonstrations, env=env,
    runner_cfg=runner_cfg, device=DEVICE,
)
print(f"Runner ready  obs_dim={OBS_DIM}  action_dim={ACTION_DIM}")

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="MPAIL2 Policy Server")


class ObsRequest(BaseModel):
    obs: list[float]            # flat list, length = obs_dim


class ActionResponse(BaseModel):
    action: list[float]         # flat list, length = action_dim


class TransitionRequest(BaseModel):
    obs:      list[float]
    action:   list[float]
    reward:   float
    next_obs: list[float]
    done:     bool


@app.get("/health")
def health():
    return {"status": "ok", "obs_dim": OBS_DIM, "action_dim": ACTION_DIM}


@app.post("/act", response_model=ActionResponse)
def act(req: ObsRequest):
    """
    Takes a flat observation vector, returns a flat action vector.
    Called by the LeRobot actor at every timestep.
    """
    obs_t = torch.tensor(req.obs, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)
    obs_dict = {OBS_KEY: obs_t}
    with torch.no_grad():
        action = runner.learner.act(obs_dict)   # (1, action_dim)
    return ActionResponse(action=action.squeeze(0).tolist())


@app.post("/step_done")
def step_done(req: TransitionRequest):
    """
    Tells MPAIL2 about the transition that just happened so it can
    store it and (periodically) run a learner update.
    """
    obs_dict      = {OBS_KEY: torch.tensor(req.obs,      dtype=torch.float32).unsqueeze(0)}
    next_obs_dict = {OBS_KEY: torch.tensor(req.next_obs, dtype=torch.float32).unsqueeze(0)}
    reward        = torch.tensor([req.reward], dtype=torch.float32)
    done          = torch.tensor([int(req.done)], dtype=torch.long)

    runner.learner.process_env_step(reward, done, {}, next_obs_dict)
    return {"status": "ok"}


@app.post("/reset")
def reset_planner():
    """Call this when the episode resets."""
    runner.learner.planner.reset()
    runner.learner.storage.clear()   # ← fix: clear rollout buffer between episodes
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
