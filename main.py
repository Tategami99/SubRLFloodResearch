from environment import FloodCoverageEnvironment
from network import SensorPlacementPolicy
from visualizer import TrainingVisualizer
from trainer import SubPolicyTrainer
import numpy as np
import time
import pygame
from pygame.locals import *

def create_density_map():
    map_size = 8
    density_map = np.zeros((map_size, map_size))
    density_map[2:4, 2:4] = 0.8
    density_map[5:7, 1:3] = 0.6 
    density_map[1:3, 5:7] = 0.7
    density_map[0:2, 0:2] = 0.4
    density_map[6:8, 6:8] = 0.3
    return density_map

def main():
    density_map = create_density_map()
    env = FloodCoverageEnvironment(density_map, n_drones=5, coverage_radius=1)
    
    cell_size = 40
    env_width = density_map.shape[1] * cell_size
    env_height = density_map.shape[0] * cell_size
    
    display_width = 1920//2
    display_height = 1080//2
    visualizer = TrainingVisualizer(env_width, env_height, cell_size, display_width, display_height)
    
    input_dim = density_map.size
    n_actions = len(env.possible_locations)
    policy_network = SensorPlacementPolicy(input_dim, n_actions, hidden_dim=128)
    
    trainer = SubPolicyTrainer(env, policy_network, learning_rate=0.0005, entropy_coef=0.01)
    
    running = True
    paused = False
    epochs = 1000
    batch_size = 8
    
    print("Starting SubPo Training with Live Visualization")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Pause/Resume training")
    print("  S - Save current model")
    print("  E - Export training data to CSV") 
    print("  ESC - Stop training")
    print("=" * 50)
    
    # store the last good state for pause visualization
    last_good_state = None
    current_epoch = 0
    
    while running and current_epoch < epochs:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == K_s:
                    trainer.save_model(f"./data/subpo_epoch_{current_epoch}.pth")
                elif event.key == K_e:
                    trainer.export_training_data(f"./data/training_epoch_{current_epoch}.csv")
                elif event.key == K_ESCAPE:
                    running = False
        
        if paused:
            # when paused, render the last good state but don't train
            if last_good_state:
                env_state, _ = last_good_state
                # restore environment state for visualization
                env.drone_positions = env_state['drone_positions']
                env.covered = env_state['covered']
                
                training_data = trainer.get_training_data()
                running = visualizer.render(
                    env, current_epoch, 
                    env_state['current_reward'], 
                    env_state['current_marginal'], 
                    env_state['current_dup_rate'],
                    training_data,
                    paused = True
                )
            else:
                # if no good state yet, just render current env
                training_data = trainer.get_training_data()
                running = visualizer.render(
                    env, current_epoch, 0, 0, 0, training_data, paused=True
                )
            
            time.sleep(0.1)
            continue  # skip the rest of the training loop when paused
            
        # TRAINING LOGIC
        # collect trajectories and get visualization data
        (batch_trajectories, epoch_rewards, epoch_marginal_sums, 
         epoch_duplicate_rates, viz_data) = trainer.collect_trajectories(batch_size, use_action_masking=True)
        
        # update policy
        loss = trainer.update_policy(batch_trajectories)
        
        # update metrics
        avg_reward = np.mean(epoch_rewards)
        avg_marginal_sum = np.mean(epoch_marginal_sums)
        avg_duplicate_rate = np.mean(epoch_duplicate_rates)
        best_reward = trainer.update_metrics(current_epoch, avg_reward, avg_marginal_sum, avg_duplicate_rate, loss)
        
        # print progress
        # if current_epoch % 10 == 0:
        #     print(f"Epoch {current_epoch:3d}: Reward={avg_reward:6.2f}, "
        #           f"Marginal={avg_marginal_sum:6.2f}, Loss={loss:7.3f}, "
        #           f"Duplicates={avg_duplicate_rate:5.1f}%")
        
        # visualize first trajectory of the batch
        if viz_data:
            first_trajectory = viz_data[0]
            current_reward = epoch_rewards[0]
            current_marginal = epoch_marginal_sums[0]
            current_dup_rate = epoch_duplicate_rates[0]
            
            # update environment state for visualization
            env.drone_positions = first_trajectory['drone_positions']
            env.covered = first_trajectory['covered']
            
            # store state for pause visualization
            last_good_state = (
                {
                    'drone_positions': first_trajectory['drone_positions'].copy(),
                    'covered': first_trajectory['covered'].copy(),
                    'current_reward': current_reward,
                    'current_marginal': current_marginal,
                    'current_dup_rate': current_dup_rate
                },
                viz_data
            )
            
            # get training data for graphs
            training_data = trainer.get_training_data()
            
            # render everything
            running = visualizer.render(
                env, current_epoch, current_reward, current_marginal, current_dup_rate, training_data
            )
        
        # slow down for visualization
        time.sleep(0.1)
        current_epoch += 1  # only increment epoch when not paused
    
    print("Training completed!")

if __name__ == "__main__":
    main()