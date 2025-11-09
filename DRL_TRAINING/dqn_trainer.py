# DRL_TRAINING/dqn_trainer.py
# (This is the complete, UPDATED script with a Learning Rate Scheduler)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import random
from collections import deque, namedtuple

# Import our custom environment, model, and opponent league
from tron_env import TronEnv
from model import QNetwork # This now imports the Dueling model
from minimax_opponents import get_hybrid_move, get_aggressive_move, get_voronoi_move
from case_closed_game import GameResult

# --- Hyperparameters ---
LEARNING_RATE = 1e-4        # STARTING learning rate
LR_END = 1e-6               # FINAL learning rate to decay to
TOTAL_TIMESTEPS = 2_000_000 # Total steps to train for
BUFFER_SIZE = 100_000       # Max experiences to store in replay buffer
BATCH_SIZE = 64             # Samples to train on each update
GAMMA = 0.99                # Discount factor for future rewards
TAU = 0.005                 # For soft updating the target network
LEARN_START = 10_000        # Steps to take before starting training
TRAIN_FREQUENCY = 4         # How often to run a training update (every 4 steps)

# Epsilon-greedy exploration parameters
EPSILON_START = 1.0         # 100% random actions at the start
EPSILON_END = 0.05          # 5% random actions at the end
EPSILON_DECAY_STEPS = 500_000 # How long to decay from START to END

SAVE_PATH = "tron_dd_dqn_model.pth" # New model name

# --- Replay Buffer ---
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

# --- Epsilon Decay Function ---
def get_epsilon(step):
    fraction = min(1.0, step / EPSILON_DECAY_STEPS)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)

# --- Main Training Function ---
def main():
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    writer = SummaryWriter(f"runs/tron_dd-dqn_run_{int(time.time())}")

    opponent_league = [get_hybrid_move, get_aggressive_move, get_voronoi_move]
    env = TronEnv(opponent_move_funcs=opponent_league)
    
    state_size = env.state_space_n
    action_size = env.action_space_n
    
    # --- 2. Initialize Models ---
    q_network = QNetwork(state_size, action_size).to(device)
    target_network = QNetwork(state_size, action_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    
    # --- NEW: Setup the Learning Rate Scheduler ---
    # We will decay the LR linearly over the total number of *optimizer updates*
    total_updates = (TOTAL_TIMESTEPS - LEARN_START) // TRAIN_FREQUENCY
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=LR_END / LEARNING_RATE, # Calculate the fractional end
        total_iters=total_updates
    )
    # --- END NEW ---
    
    replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
    
    # --- 3. Training Loop ---
    print("Starting DD-DQN training with LR Scheduler...")
    start_time = time.time()
    state = env.reset()
    
    for global_step in range(1, TOTAL_TIMESTEPS + 1):
        
        # --- 3.A. Epsilon-Greedy Action Selection ---
        epsilon = get_epsilon(global_step)
        if random.random() < epsilon:
            action = random.randrange(action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.Tensor(state).to(device).unsqueeze(0)
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()
        
        # --- 3.B. Step the Environment ---
        next_state, reward, done, info = env.step(action)
        
        # --- 3.C. Store Experience ---
        replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        
        # --- 3.D. Handle Game Over ---
        if done:
            result = info.get("result", GameResult.DRAW)
            win = 1 if result == GameResult.AGENT1_WIN else 0
            loss = 1 if result == GameResult.AGENT2_WIN else 0
            draw = 1 if result == GameResult.DRAW else 0
            
            writer.add_scalar("charts/Win", win, global_step)
            writer.add_scalar("charts/Loss", loss, global_step)
            writer.add_scalar("charts/Draw", draw, global_step)
            writer.add_scalar("charts/Game_Length", env.game.turns, global_step)
            
            state = env.reset()

        # --- 3.E. Training Update ---
        if global_step > LEARN_START and global_step % TRAIN_FREQUENCY == 0:
            
            states, actions, rewards, next_states, dones = replay_buffer.sample()
            
            # --- Double DQN Logic ---
            with torch.no_grad():
                next_q_values_main = q_network(next_states)
                best_action_indices = next_q_values_main.argmax(dim=1, keepdim=True)
                next_q_values_target = target_network(next_states)
                max_next_q = next_q_values_target.gather(1, best_action_indices)
                target_q = rewards + GAMMA * max_next_q * (1 - dones)

            # --- Calculate Current Q-Values ---
            current_q_all = q_network(states)
            current_q = current_q_all.gather(1, actions)
            
            # --- Calculate Loss ---
            loss = F.smooth_l1_loss(current_q, target_q)
            writer.add_scalar("losses/q_loss", loss.item(), global_step)
            
            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
            optimizer.step()
            
            # --- NEW: Step the scheduler ---
            scheduler.step()

            # --- Soft Update Target Network ---
            for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(TAU * q_param.data + (1.0 - TAU) * target_param.data)
        
        # --- 4. Logging ---
        writer.add_scalar("charts/Epsilon", epsilon, global_step)
        
        # --- NEW: Log the learning rate ---
        if global_step % 1000 == 0: # Log LR every 1000 steps
            writer.add_scalar("charts/Learning_Rate", optimizer.param_groups[0]["lr"], global_step)

        if global_step % 10000 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"Step: {global_step}/{TOTAL_TIMESTEPS}, SPS: {sps}, Epsilon: {epsilon:.3f}, LR: {optimizer.param_groups[0]['lr']:.8f}")
            writer.add_scalar("charts/SPS", sps, global_step)
            
            torch.save(q_network.state_dict(), SAVE_PATH.replace(".pth", f"_ckpt.pth"))
            
    # --- 5. Final Save ---
    env.close()
    writer.close()
    torch.save(q_network.state_dict(), SAVE_PATH)
    print(f"Training complete. Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()