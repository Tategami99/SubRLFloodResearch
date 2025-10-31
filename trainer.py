import torch
import numpy as np

class SubPolicyTrainer:
    def __init__(self, env, policy_network, learning_rate=0.001):
        self.env = env
        self.policy_network = policy_network
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # training metrics
        self.epochs = []
        self.epoch_rewards = []
        self.epoch_losses = []
        self.epoch_marginals = []
        self.best_rewards = []
        self.duplicate_rates = []
    
    def compute_duplicate_rate(self, drone_positions):
        if not drone_positions:
            return 0.0
        unique = len(set(drone_positions))
        return ((len(drone_positions) - unique) / len(drone_positions)) * 100
    
    # use marginal gains to compute submodular policy gradient(theorem 2)
    def compute_subpo_gradient(self, trajectories):
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

            # normalize advantages
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # compute policy gradient
            log_probs = []
            for state, action in zip(states, actions):
                state_tensor = torch.tensor(state, dtype=torch.float32)
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
    
    # collect trajectories for one epoch - returns metrics for visualization
    def collect_trajectories(self, batch_size=8):
        batch_trajectories = []
        epoch_rewards = []
        epoch_marginal_sums = []
        epoch_duplicate_rates = []
        all_states_actions = []

        for i in range(batch_size):
            state = self.env.reset()
            done = False
            trajectory = {'states': [], 'actions': [], 'marginal_gains': [], 'rewards': []}

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action_probs = self.policy_network(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()

                next_state, marginal_gain, total_reward, done = self.env.step(action.item())

                trajectory['states'].append(state)
                trajectory['actions'].append(action.item())
                trajectory['marginal_gains'].append(marginal_gain)
                trajectory['rewards'].append(total_reward)

                state = next_state

            batch_trajectories.append((trajectory['states'], trajectory['actions'], trajectory['marginal_gains']))
            epoch_rewards.append(trajectory['rewards'][-1])
            epoch_marginal_sums.append(sum(trajectory['marginal_gains']))
            epoch_duplicate_rates.append(self.compute_duplicate_rate(self.env.drone_positions))
            
            # store for visualization
            all_states_actions.append({
                'states': trajectory['states'],
                'actions': trajectory['actions'],
                'drone_positions': self.env.drone_positions.copy(),
                'covered': self.env.covered.copy()
            })

        return batch_trajectories, epoch_rewards, epoch_marginal_sums, epoch_duplicate_rates, all_states_actions
    
    # update policy with collected trajectories
    def update_policy(self, batch_trajectories):
        self.optimizer.zero_grad()
        loss = self.compute_subpo_gradient(batch_trajectories)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_metrics(self, epoch, avg_reward, avg_marginal_sum, avg_duplicate_rate, loss):
        if avg_reward > (self.best_rewards[-1] if self.best_rewards else -float('inf')):
            best_reward = avg_reward
        else:
            best_reward = self.best_rewards[-1] if self.best_rewards else avg_reward

        self.epochs.append(epoch)
        self.epoch_rewards.append(avg_reward)
        self.epoch_losses.append(loss)
        self.epoch_marginals.append(avg_marginal_sum)
        self.best_rewards.append(best_reward)
        self.duplicate_rates.append(avg_duplicate_rate)
        
        return best_reward
    
    def get_training_data(self):
        """Get current training data for visualization - FIXED FORMAT"""
        return (
            self.epochs,
            self.epoch_rewards, 
            self.epoch_losses,
            self.epoch_marginals,
            self.duplicate_rates,
            self.best_rewards
        )
    
    def save_model(self, filename="subpo_model.pth"):
        torch.save({
            'model_state_dict': self.policy_network.state_dict(),
            'training_data': {
                'epochs': self.epochs,
                'rewards': self.epoch_rewards,
                'losses': self.epoch_losses, 
                'marginals': self.epoch_marginals,
                'duplicates': self.duplicate_rates,
                'best_rewards': self.best_rewards
            }
        }, filename)
        print(f"Model saved to {filename}")
    
    def export_training_data(self, filename="training_data.csv"):
        if len(self.epochs) == 0:
            return
        with open(filename, 'w') as f:
            f.write("Epoch,Average_Reward,Best_Reward,Loss,Marginal_Gains,Duplicate_Rate\n")
            for i, epoch in enumerate(self.epochs):
                f.write(f"{epoch},{self.epoch_rewards[i]:.4f},{self.best_rewards[i]:.4f},"
                       f"{self.epoch_losses[i]:.6f},{self.epoch_marginals[i]:.4f},"
                       f"{self.duplicate_rates[i]:.2f}\n")
        print(f"Data exported to {filename}")