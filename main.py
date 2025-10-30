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

    def compute_subpo_gradient(self, trajectories):
        # use marginal gains to compute submodular policy gradient(theorem 2)
        policy_gradients = []

        for trajectory in trajectories:
            states, actions, marginal_gains, baselines = trajectory

            cumulative_marginal_gains = []
            R = 0
            for marginal_gain in reversed(marginal_gains):
                R += marginal_gain
                cumulative_marginal_gains.insert(0, R)

            cumulative_marginal_gains = torch.tensor(cumulative_marginal_gains, dtype=torch.float32)

            # normalize advantages
            advantages = cumulative_marginal_gains - torch.tensor(baselines, dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

            # compute policy gradient with marginal gains
            log_probs = []
            for state, action in zip(states, actions):
                state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
                action_probs = self.policy_network(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(torch.tensor(action))
                log_probs.append(log_prob)
            
            # submodular policy gradient: sum of log_prob * advantage
            loss = 0
            for log_prob, advantage in zip(log_probs, advantages):
                loss -= log_prob * advantage

            policy_gradients.append(loss)

        return torch.stack(policy_gradients).mean()
        
    def train(self, epochs=100, batch_size=8, entropy_coef=0.01):
        for epoch in range(epochs):
            batch_trajectories = []
            epoch_rewards = []

            #collect trajectories
            for _ in range(batch_size):
                state = self.env.reset()
                done = False
                trajectory = {
                    'states': [],
                    'actions': [],
                    'marginal_gains': [],
                    'baselines': [],
                    'rewards': []
                }

                step_count = 0
                while not done:
                    state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
                    action_probs = self.policy_network(state_tensor)

                    # add entropy for exploration
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    next_state, marginal_gain, total_reward, done = self.env.step(action.item())

                    # simple baseline: average marginal gain so far
                    baseline = np.mean(trajectory['marginal_gains']) if trajectory['marginal_gains'] else 0.0

                    trajectory['states'].append(state)
                    trajectory['actions'].append(action.item())
                    trajectory['marginal_gains'].append(marginal_gain)
                    trajectory['baselines'].append(baseline)
                    trajectory['rewards'].append(total_reward)

                    state = next_state
                    step_count += 1

                    # render occasionally
                    if epoch % 10 == 0 and batch_size == 0:
                        self.env.render()
                        time.sleep(0.1)

                batch_trajectories.append((
                    trajectory['states'],
                    trajectory['actions'],
                    trajectory['marginal_gains'],
                    trajectory['baselines']
                ))
                epoch_rewards.append(sum(trajectory['rewards']))

            # update policy using subpo gradient
            self.optimizer.zero_grad()
            loss = self.compute_subpo_gradient(batch_trajectories)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                avg_reward = np.mean(epoch_rewards)
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.4f}")


def train():
    density_map = np.array([
        [0.1, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.5, 0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.6, 0.8, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.3, 0.5, 0.6, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.3, 0.5, 0.6, 0.5, 0.3, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.3, 0.5, 0.6, 0.5, 0.3, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.6, 0.5, 0.3, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.6, 0.5, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.6, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.6]
    ])

    env = FloodCoverageEnvironment(density_map, n_drones=5, coverage_radius=1)

    input_dim = density_map.size #total grid cells
    n_actions = len(env.possible_locations) #possible drone placements
    policy_network = SensorPlacementPolicy(input_dim, n_actions)
    trainer = SubPolicyTrainer(env, policy_network, learning_rate=0.001)
    trainer.train(epochs=100, batch_size=8)

    env.close()  
    

if __name__ == "__main__":
    train()