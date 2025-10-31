import pygame
import matplotlib.pyplot as plt
import io
import numpy as np
import time

class TrainingVisualizer:
    def __init__(self, env_width, env_height, cell_size, display_width=1400, display_height=900):
        self.env_width = env_width
        self.env_height = env_height
        self.cell_size = cell_size
        
        self.display_width = display_width
        self.display_height = display_height
        
        self.display = None
        self.graph_surface = None
        self.last_plot_time = 0
        self.plot_interval = 2  # seconds between plot updates
        
        pygame.init()
        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("SubPo Training - Flood Coverage")
        
        base_font_size = max(18, display_width // 70)
        self.font = pygame.font.Font(None, base_font_size)
        self.small_font = pygame.font.Font(None, base_font_size - 4)
        self.large_font = pygame.font.Font(None, base_font_size + 4)
        self.title_font = pygame.font.Font(None, base_font_size + 8)
    
    def create_graph_surface(self, epochs, rewards, losses, marginals, duplicates, best_rewards):
        if len(epochs) < 2:  # need at least 2 points to plot
            return None
            
        try:
            # calculate figure size based on display dimensions
            fig_width = 8
            fig_height = 6
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(fig_width, fig_height))
            
            # Plot 1: Rewards
            ax1.plot(epochs, rewards, 'b-', alpha=0.7, linewidth=2, label='Avg Reward')
            ax1.plot(epochs, best_rewards, 'r-', linewidth=2, label='Best Reward')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Coverage Reward')
            ax1.set_title('Training Rewards')
            ax1.legend(fontsize='small')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Loss
            ax2.plot(epochs, losses, 'g-', alpha=0.7, linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Policy Loss')
            ax2.set_title('Policy Gradient Loss')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Marginal gains
            ax3.plot(epochs, rewards, 'b-', linewidth=2, label='Total Rewards')
            ax3.plot(epochs, marginals, 'orange', linewidth=2, label='Marginal Gains')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Value')
            ax3.set_title('Rewards vs Marginal Gains')
            ax3.legend(fontsize='small')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Duplicate rate
            ax4.plot(epochs, duplicates, 'red', linewidth=2, marker='o', markersize=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Duplicate Rate (%)')
            ax4.set_title('Duplicate Placement Rate')
            max_dup = max(duplicates) if duplicates else 100
            ax4.set_ylim(0, max(100, max_dup * 1.1))
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout(pad=2.0)
            
            # Convert to Pygame surface
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plot_image = pygame.image.load(buf)
            plt.close()
            
            return plot_image
        except Exception as e:
            print(f"Error creating graph: {e}")
            return None
    
    def render(self, env, epoch, current_reward, current_marginal, duplicate_rate, 
               training_data=None, paused=False):
        # clear display
        self.display.fill((30, 30, 30))
        
        # calculate layout
        env_panel_width = self.display_width * 4 // 10  # 40% for environment
        graph_panel_width = self.display_width * 6 // 10  # 60% for graphs
        metrics_height = 120  # height for metrics at bottom
        
        env_panel_height = self.display_height - metrics_height
        graph_panel_height = self.display_height - metrics_height
        
        # ===== ENVIRONMENT PANEL (LEFT) =====
        env_title = self.title_font.render("Placement Map", True, (255, 255, 255))
        self.display.blit(env_title, (20, 10))
        
        # scale environment to fit panel
        env_scale_factor = min(
            (env_panel_width - 40) / self.env_width,
            (env_panel_height - 40) / self.env_height
        )
        scaled_env_width = int(self.env_width * env_scale_factor)
        scaled_env_height = int(self.env_height * env_scale_factor)
        
        # render environment
        env_surface = pygame.Surface((self.env_width, self.env_height))
        env_surface.fill((255, 255, 255))
        
        # draw environment grid
        for x in range(env.density_map.shape[0]):
            for y in range(env.density_map.shape[1]):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                 self.cell_size, self.cell_size)
                density = env.density_map[x, y]
                if env.covered[x, y]:
                    color = (0, min(255, 100 + int(155 * density)), 0)
                else:
                    color = (min(255, 100 + int(155 * density)), 0, 0)
                pygame.draw.rect(env_surface, color, rect)
                pygame.draw.rect(env_surface, (50, 50, 50), rect, 1)
        
        # draw drones
        for pos in env.drone_positions:
            x, y = pos
            center = (y * self.cell_size + self.cell_size // 2, 
                     x * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(env_surface, (0, 0, 255), center, self.cell_size // 3)
            pygame.draw.circle(env_surface, (255, 255, 255), center, self.cell_size // 3, 2)
        
        # scale and position environment
        scaled_env = pygame.transform.smoothscale(env_surface, (scaled_env_width, scaled_env_height))
        env_x = 20
        env_y = 50  # Below title
        self.display.blit(scaled_env, (env_x, env_y))
        
        # environment data panel
        env_data_y = env_y + scaled_env_height + 10
        env_data = [
            f"Map Size: {env.density_map.shape[1]}x{env.density_map.shape[0]}",
            f"Coverage Radius: {env.coverage_radius}",
            f"Drones: {len(env.drone_positions)}/{env.n_drones}",
            f"Current Reward: {current_reward:.2f}",
            f"Marginal Gain: {current_marginal:.2f}",
            f"Duplicate Rate: {duplicate_rate:.1f}%"
        ]
        
        for i, data in enumerate(env_data):
            text = self.font.render(data, True, (200, 200, 200))
            self.display.blit(text, (env_x, env_data_y + i * 25))
        
        # ===== GRAPHS PANEL (RIGHT) =====
        graph_title = self.title_font.render("Training Progress", True, (255, 255, 255))
        self.display.blit(graph_title, (env_panel_width + 20, 10))
        
        # update graphs periodically
        current_time = time.time()
        if (training_data and 
            current_time - self.last_plot_time > self.plot_interval):
            self.graph_surface = self.create_graph_surface(*training_data)
            self.last_plot_time = current_time
        
        if self.graph_surface:
            graph_x = env_panel_width + 20
            graph_y = 50
            
            # scale graph to fit the allocated space
            scaled_graph = pygame.transform.smoothscale(
                self.graph_surface, 
                (graph_panel_width - 40, graph_panel_height - 60)
            )
            self.display.blit(scaled_graph, (graph_x, graph_y))
        
        # ===== METRICS PANEL (BOTTOM) =====
        metrics_bg = pygame.Rect(0, self.display_height - metrics_height, 
                               self.display_width, metrics_height)
        pygame.draw.rect(self.display, (40, 40, 40), metrics_bg)
        pygame.draw.rect(self.display, (80, 80, 80), metrics_bg, 2)
        
        metrics_y = self.display_height - metrics_height + 15
        
        # current epoch and status
        status_color = (255, 50, 50) if paused else (50, 255, 50)
        status_text = "PAUSED" if paused else "TRAINING"
        status_surface = self.large_font.render(f"Epoch: {epoch} - {status_text}", True, status_color)
        self.display.blit(status_surface, (20, metrics_y))
        
        # controls info
        controls_text = "SPACE: Pause/Resume  |  S: Save Model  |  E: Export Data  |  ESC: Quit"
        controls_surface = self.small_font.render(controls_text, True, (150, 150, 255))
        self.display.blit(controls_surface, (20, metrics_y + 35))
        
        # training summary (if available)
        if training_data and len(training_data[0]) > 0:
            epochs_list, rewards, losses, marginals, duplicates, best_rewards = training_data
            if rewards:
                summary_text = f"Best Reward: {max(rewards):.2f}  |  Avg Loss: {np.mean(losses[-10:]):.3f}  |  Avg Duplicates: {np.mean(duplicates[-10:]):.1f}%"
                summary_surface = self.small_font.render(summary_text, True, (200, 200, 100))
                self.display.blit(summary_surface, (20, metrics_y + 60))
        
        pygame.display.flip()
        return True
    
    def close(self):
        pygame.quit()