import torch
import numpy as np
from utils import create_action_mask, apply_action_mask

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
    
    #how often we visit the same location multiple times
    #submodular rewards naturally discourage duplication due to diminishing returns
    def compute_duplicate_rate(self, drone_positions):
        if not drone_positions:
            return 0.0
        unique = len(set(drone_positions))
        return ((len(drone_positions) - unique) / len(drone_positions)) * 100
    
    """
    Regular Policy Gradient (REINFORCE):
    ∇J(θ) = E[∑ ∇logπ(a|s) * (total future reward)]

    Submodular Policy Gradient (SubPo):
    ∇J(θ) = E[∑ ∇logπ(a|s) * (∑ future MARGINAL GAINS)]

    use marginal gains instead of rewards
    focuses on actions that provide new value instead of
    just high-rewarding actions
    """
    def compute_subpo_gradient(self, trajectories, entropy_coef):
        policy_gradients = []

        for trajectory in trajectories:
            states, actions, marginal_gains = trajectory

            if len(marginal_gains) == 0:
                continue

            H = len(marginal_gains)
            cumulative_future_gains = []

            #cumulative marginal gain
            # ∑_{j=i}^{H-1} F(s_{j+1}|τ_{0:j})
            # for each time step, compute the sum of future marginal gains
            for i in range(H):
                discount_factor = 0.95 #how much we value immediate vs future gains
                future_gain = sum(marginal_gains[i] * (discount_factor ** (j - i)) for j in range(i, H))
                cumulative_future_gains.append(future_gain)

            cumulative_future_gains = torch.tensor(cumulative_future_gains, dtype=torch.float32)

            # simple baseline: average marginal gain
            # baseline = cumulative_future_gains.mean()
            # advantages = cumulative_future_gains - baseline

            #subtract a baseline b(τ_{0:i}) from the score gradient
            # better baseline: moving average of marginal gains

            """
            VARIANCE REDUCTION WITH BASELINES
            
            raw cumulative gains can have high variance which makes training unstable

            subtract a baseline(moving average) from the score gradient
            to center the values
            reduces variance without changing the expected gradient
            """
            baseline = torch.zeros_like(cumulative_future_gains)
            for i in range(H):
                if i == 0:
                    baseline[i] = cumulative_future_gains[i]
                else:
                    #exponential moving average: 80% previous baseline + 20% current value
                    baseline[i] = 0.8 * baseline[i-1] + 0.2 * cumulative_future_gains[i]

            #how much better than expected
            advantages = cumulative_future_gains - baseline

            """
            NORMALIZE FOR STABILITY

            gradient updates depend on the scale of advantages
            if advntages are large -> large gradient steps -> training instability
            if advantages are small -> small gradient steps -> slow learning

            normalizing makes the scale consistent across different episodes
            """
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            """
            COMPUTE POLICY GRADIENT
            ∇J(π_θ) = E[∑ ∇logπ(a_i|s_i) * (∑_{j=i}^{H-1} F(s_{j+1}|τ_{0:j}) - b(τ_{0:i}))]

            ∇logπ(a_i|s_i) is the score function (how changing θ affects action probability)
            the sum term is our "advantages" (how good this action was for future marginal gains)

            b(τ_{0:i}) is our baseline (what we expected to get)
            """
            total_loss = 0.0
            entropy_sum = 0.0
            for i, (state, action) in enumerate(zip(states, actions)):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action_probs = self.policy_network(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                log_prob = dist.log_prob(torch.tensor(action)) #ln(probability of chosen action)
                entropy = dist.entropy() #measure of randomness(higher = more exploration)

                """
                POLICY GRADIENT TERM:
                log_prob * advantage means:
                increase probability of actions that had positive advantage (better than expected)"
                decrease probability of actions that had negative advantage (worse than expected)"
                """
                policy_loss = -log_prob * advantages[i]

                """
                ENTROPY REGULARIZATION:
                higher entropy -> more random actions -> more exploration
                add entropy to encourage exploration(negative sign to miniimize loss)
                """
                entropy_loss = -entropy_coef * entropy

                total_loss += policy_loss + entropy_loss
                entropy_sum += entropy
            
            policy_gradients.append(total_loss / len(actions))

        return torch.stack(policy_gradients).mean() if policy_gradients else torch.tensor(0.0)
    
    """
    collect trajectories for one epoch - returns metrics for visualization

    COLLECT EXPERIENCE BY RUNNING THE CURRENT POLICY
    Sample a_h ∼ π(a_h|s_h), execute a_h and collect data
    """
    def collect_trajectories(self, batch_size=8, use_action_masking=False):
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

                #sample from the probability distribution
                #sometimes take suboptimal actions
                if use_action_masking:
                    #prevents selecting already-visited positions   
                    action_mask = create_action_mask(self.env.visited_locations, self.env.possible_locations)
                    masked_probs = apply_action_mask(action_probs, action_mask)
                    dist = torch.distributions.Categorical(masked_probs)
                else:
                    dist = torch.distributions.Categorical(action_probs)
                    
                action = dist.sample()

                next_state, marginal_gain, total_reward, done = self.env.step(action.item())

                trajectory['states'].append(state)
                trajectory['actions'].append(action.item())
                trajectory['marginal_gains'].append(marginal_gain)
                trajectory['rewards'].append(total_reward)

                state = next_state

            #store complete trajectory for gradient computation
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
    def update_policy(self, batch_trajectories, entropy_coef):
        if not batch_trajectories or len(batch_trajectories) == 0:
            return 0.0

        self.optimizer.zero_grad()

        #compute loss
        loss = self.compute_subpo_gradient(batch_trajectories, entropy_coef)
        
        # only backward if valid loss
        if loss != 0.0:
            loss.backward() #compute gradients using backpropogation
            # if gradeints get too large, scale them to a max norm of 1
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
            self.optimizer.step() #update network weights

        # if len(self.epoch_rewards) > 200:
        #     recent_best = max(self.epoch_rewards[-100:])
        #     previous_best = max(self.epoch_rewards[-200:-100])
        #     if recent_best - previous_best < 0.05: # very little improvement
        #         for param_group in self.optimizer.param_groups:
        #             param_group['lr'] *= 0.8  # reduce learning rate

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