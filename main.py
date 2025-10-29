from environment import FloodCoverageEnvironment
from network import SensorPlacementPolicy
import torch
import numpy as np
import time

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

    optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.001)
    
    epochs = 10
    batch_size = 4 #TODO: increase for actual training

    for epoch in range(epochs):
        batch_losses = []
        rewards_per_epoch = []

        for batch in range(batch_size):
            #start new episode
            state, _ = env.reset()
            done = False
            log_probabilities = []
            rewards = []
            env.render()

            while not done:
                #Prepare state for neural network
                state_tensor = torch.tensor(state.flatten(), dtype=torch.float32) #flatten for input
                action_probabilities = policy_network(state_tensor)

                dist = torch.distributions.Categorical(action_probabilities)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                #get rewards and next state from the action taken
                next_state, reward, done = env.step(action.item()) 
                log_probabilities.append(log_prob)
                rewards.append(reward)

                env.render()
                time.sleep(0.1)

                state = next_state

            # Compute discounted returns
            discounted_rewards = []
            R = 0
            gamma = 0.99
            for r in reversed(rewards):
                R = r + gamma * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            # Compute loss (policy gradient)
            loss = 0
            for log_prob, R in zip(log_probabilities, discounted_rewards):
                loss -= log_prob * R #Maximize expected return

            batch_losses.append(loss)
            rewards_per_epoch.append(sum(rewards))
        
        #Gradient calculation and update
        optimizer.zero_grad()
        total_loss = torch.stack(batch_losses).mean()
        total_loss.backward()
        optimizer.step()

    env.close()        
    

if __name__ == "__main__":
    train()