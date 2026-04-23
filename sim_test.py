"""
sim_test.py
Runs the MPAIL2 + LeRobot integration loop in MuJoCo simulation.

Prerequisites:
    Terminal 1 (mpail2 env):   python ~/projects/mpail2/mpail2/policy_server.py
    Terminal 2 (lerobot env):  python sim_test.py
"""

import gymnasium as gym
import numpy as np
from lerobot.policies.mpail2_client import MPAIL2ClientPolicy

# ── config ────────────────────────────────────────────────────────────────────
ENV_ID      = "Ant-v5"
NUM_EPISODES = 3
SERVER_URL  = "http://localhost:8765"

# ── connect to MPAIL2 server ──────────────────────────────────────────────────
policy = MPAIL2ClientPolicy(server_url=SERVER_URL)

# ── create MuJoCo env ─────────────────────────────────────────────────────────
env = gym.make(ENV_ID)

# ── run episodes ──────────────────────────────────────────────────────────────
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    policy.reset()

    ep_reward = 0.0
    done      = False
    step      = 0

    print(f"\nEpisode {ep + 1}/{NUM_EPISODES}")

    while not done and step < 1000:
        # 1. get action from MPAIL2
        action = policy.select_action(obs)

        # 2. step the simulator
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. send transition back to MPAIL2 so it can learn
        policy.report_transition(obs, action, reward, next_obs, done)

        ep_reward += reward
        obs        = next_obs
        step      += 1

    print(f"  steps={step}  total_reward={ep_reward:.2f}")

env.close()
print("\nSimulation test complete.")
