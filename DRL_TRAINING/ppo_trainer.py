# DRL_TRAINING/ppo_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

# Import our custom environment and model
from tron_env import TronEnv
from model import ActorCritic

# Import the "League of Opponents"
from minimax_opponents import get_hybrid_move, get_aggressive_move, get_voronoi_move

# --- PPO Hyperparameters ---
LEARNING_RATE = 2.5e-4
NUM_ENVS = 1 # For simplicity, we'll run 1 env. Can be parallelized later.
NUM_STEPS = 2048 # Steps per policy update (e.g., 2048)
TOTAL_TIMESTEPS = 5_000_000 # Total steps to train for
UPDATE_EPOCHS = 10 # Num of epochs to update policy per batch
MINIBATCH_SIZE = 64
GAMMA = 0.99 # Discount factor
GAE_LAMBDA = 0.95 # Generalized Advantage Estimation lambda
CLIP_EPS = 0.2 # PPO clip range
ENT_COEFF = 0.01 # Entropy bonus coefficient
VF_COEFF = 0.5 # Value function loss coefficient
MAX_GRAD_NORM = 0.5
SAVE_PATH = "tron_model.pth"

def main():
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup TensorBoard for logging
    writer = SummaryWriter(f"runs/tron_ppo_run_{int(time.time())}")

    # Instantiate the "League of Opponents"
    opponent_league = [get_hybrid_move, get_aggressive_move, get_voronoi_move]
    
    # Instantiate the environment
    env = TronEnv(opponent_move_funcs=opponent_league)
    
    state_size = env.state_space_n
    action_size = env.action_space_n
    
    # Instantiate the agent
    agent = ActorCritic(state_size, action_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # --- 2. PPO Storage ---
    # These will store the experiences from our rollouts
    states = torch.zeros((NUM_STEPS, state_size)).to(device)
    actions = torch.zeros((NUM_STEPS,)).to(device)
    log_probs = torch.zeros((NUM_STEPS,)).to(device)
    rewards = torch.zeros((NUM_STEPS,)).to(device)
    dones = torch.zeros((NUM_STEPS,)).to(device)
    values = torch.zeros((NUM_STEPS,)).to(device)

    # --- 3. Training Loop ---
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // NUM_STEPS
    
    start_time = time.time()
    next_state = torch.Tensor(env.reset()).to(device)
    next_done = torch.zeros(1).to(device)

    for update in range(1, num_updates + 1):
        
        # --- 3.A. Rollout Phase (Collecting Experience) ---
        for step in range(0, NUM_STEPS):
            global_step += 1
            states[step] = next_state
            dones[step] = next_done

            # Get action from the agent
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_state.unsqueeze(0))
                values[step] = value.flatten()
            
            actions[step] = action
            log_probs[step] = log_prob

            # Take action in the environment
            next_state, reward, done, info = env.step(action.item())
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_state, next_done = torch.Tensor(next_state).to(device), torch.tensor(done).to(device)
            
            if done:
                # Log game results
                result = info.get("result", GameResult.DRAW)
                win = 1 if result == GameResult.AGENT1_WIN else 0
                loss = 1 if result == GameResult.AGENT2_WIN else 0
                draw = 1 if result == GameResult.DRAW else 0
                
                writer.add_scalar("charts/Win", win, global_step)
                writer.add_scalar("charts/Loss", loss, global_step)
                writer.add_scalar("charts/Draw", draw, global_step)
                writer.add_scalar("charts/Game_Length", env.game.turns, global_step)
                
                # Reset environment
                next_state = torch.Tensor(env.reset()).to(device)
                next_done = torch.zeros(1).to(device)


        # --- 3.B. GAE & Return Calculation ---
        with torch.no_grad():
            # Get value of the *last* state
            next_value = agent.forward(next_state.unsqueeze(0))[1].reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lambda = 0
            
            # Calculate advantages backwards
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                    
                delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lambda = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lambda
            
            # Calculate returns
            returns = advantages + values

        # --- 3.C. PPO Update Phase ---
        
        # Flatten the batch
        b_states = states.reshape(-1, state_size)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update policy for K epochs
        for epoch in range(UPDATE_EPOCHS):
            # Get minibatches
            b_inds = np.arange(NUM_STEPS)
            np.random.shuffle(b_inds)
            
            for start in range(0, NUM_STEPS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_states[mb_inds], b_actions[mb_inds]
                )
                
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = log_ratio.exp()

                # Advantage normalization
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # --- Policy Loss (Actor) ---
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- Value Loss (Critic) ---
                new_value = new_value.view(-1)
                v_loss = F.mse_loss(new_value, b_returns[mb_inds])
                
                # --- Entropy Loss (Exploration) ---
                entropy_loss = entropy.mean()

                # --- Total Loss ---
                loss = pg_loss - ENT_COEFF * entropy_loss + v_loss * VF_COEFF
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # --- 4. Logging ---
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        
        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update}/{num_updates}, Global Step: {global_step}, SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    # --- 5. Save Final Model ---
    env.close()
    writer.close()
    torch.save(agent.state_dict(), SAVE_PATH)
    print(f"Training complete. Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()