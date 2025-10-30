from environment import FloodCoverageEnvironment
from network import SensorPlacementPolicy
import torch
import numpy as np
import time

class SubPolicyTrainer:
    def __init__(self, env, policy_network, learning_rate=0.001):
        self.env = env
        self.policy_network = policy_network
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def visualize_policy(self, epoch):
        """Visualize the current policy"""
        state = self.env.reset()
        done = False
        total_reward = 0
        
        print(f"\n--- Epoch {epoch} Policy Visualization ---")
        
        while not done:
            # get action probabilities
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_probs = self.policy_network(state_tensor)
            
            # choose best action for visualization
            action = torch.argmax(action_probs).item()
            
            # Take action
            next_state, marginal_gain, total_reward, done = self.env.step(action)
            
            # Render
            self.env.render()
            time.sleep(0.5)  # slow down for visualization
            
            state = next_state
        
        print(f"Final coverage: {total_reward:.2f}")
        print("Drone positions:", self.env.drone_positions)
        print("---" + "-" * len(f"Epoch {epoch} Policy Visualization ---"))

    def compute_subpo_gradient(self, trajectories):
        # use marginal gains to compute submodular policy gradient(theorem 2)
        policy_gradients = []

        for trajectory in trajectories:
            states, actions, marginal_gains = trajectory

            H = len(marginal_gains)
            cumulative_future_gains = []
            for i in range(H):
                future_gain = sum(marginal_gains[i:])
                cumulative_future_gains.append(future_gain)

            cumulative_future_gains = torch.tensor(cumulative_future_gains, dtype=torch.float32)

            # simple baseline: average marginal gain
            baseline = cumulative_future_gains.mean()
            advantages = cumulative_future_gains - baseline

            #normalize advantages
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            #compute policy gradient
            log_probs = []
            for state, action in zip(states, actions):
                state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
                action_probs = self.policy_network(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(torch.tensor(action))
                log_probs.append(log_prob)

            # submodular policy gradient
            loss = 0
            for log_prob, advantage in zip(log_probs, advantages):
                loss -= log_prob * advantage
            
            policy_gradients.append(loss)

        return torch.stack(policy_gradients).mean()
        
    def train(self, epochs=100, batch_size=8, render_freq=20):
        best_reward = -float('inf')

        for epoch in range(epochs):
            batch_trajectories = []
            epoch_rewards = []
            epoch_marginal_gains = []

            #collect trajectories
            for i in range(batch_size):
                state = self.env.reset()
                done = False
                trajectory = {
                    'states': [],
                    'actions': [],
                    'marginal_gains': [],
                    'rewards': []
                }

                while not done:
                    state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
                    action_probs = self.policy_network(state_tensor)

                    # sample action
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()

                    # take action
                    if action.item() >= len(self.env.possible_locations):
                        action = torch.tensor(np.random.randint(0, len(self.env.possible_locations)))

                    next_state, marginal_gain, total_reward, done = self.env.step(action.item())

                    trajectory['states'].append(state)
                    trajectory['actions'].append(action.item())
                    trajectory['marginal_gains'].append(marginal_gain)
                    trajectory['rewards'].append(total_reward)

                    state = next_state

                batch_trajectories.append((
                    trajectory['states'],
                    trajectory['actions'],
                    trajectory['marginal_gains']
                ))
                epoch_rewards.append(trajectory['rewards'][-1]) # final total reward
                epoch_marginal_gains.append(sum(trajectory['marginal_gains']))

                # render first trajectory of the last batch occasionally
                if epoch % render_freq == 0 and i == 0:
                    self.visualize_policy(epoch)

            if batch_trajectories:
                self.optimizer.zero_grad()
                loss = self.compute_subpo_gradient(batch_trajectories)
                loss.backward()
                self.optimizer.step()

                avg_reward = np.mean(epoch_rewards)
                avg_marginal_gain = np.mean(epoch_marginal_gains)

                if avg_reward > best_reward:
                    best_reward = avg_reward

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}, Loss: {loss.item():7.4f}, "
                          f"Avg Reward: {avg_reward:6.2f}, Best: {best_reward:6.2f}, "
                          f"Avg Marginal: {avg_marginal_gain:6.2f}")

def create_density_map():
    map_size = 8
    density_map = np.zeros((map_size, map_size))

    # create high-density regions
    density_map[2:4, 2:4] = 0.8
    density_map[5:7, 1:3] = 0.6 
    density_map[1:3, 5:7] = 0.7
    
    # create medium density areas
    density_map[0:2, 0:2] = 0.4
    density_map[6:8, 6:8] = 0.3
    
    return density_map

def train():
    density_map = create_density_map()

    env = FloodCoverageEnvironment(density_map, n_drones=5, coverage_radius=1, cell_size=40)

    input_dim = density_map.size #total grid cells
    n_actions = len(env.possible_locations) #possible drone placements
    policy_network = SensorPlacementPolicy(input_dim, n_actions, hidden_dim=128)
    trainer = SubPolicyTrainer(env, policy_network, learning_rate=0.001)
    try:
        trainer.train(epochs=200, batch_size=16, render_freq=25)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        env.close()  
    

if __name__ == "__main__":
    train()