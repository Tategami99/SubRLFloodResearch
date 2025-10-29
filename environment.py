import numpy as np
import pygame

class FloodCoverageEnvironment:
    def __init__(self, density_map, n_drones, coverage_radius, cell_size=30):
        self.density_map = np.array(density_map)
        self.n_drones = n_drones
        self.coverage_radius = coverage_radius
        self.possible_locations = list(np.ndindex(self.density_map.shape))
        self.cell_size = cell_size
        self.WIDTH = self.density_map.shape[1] * cell_size
        self.HEIGHT = self.density_map.shape[0] * cell_size
        self.reset()
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Flood Coverage Environment")

    def reset(self):
        self.drone_positions = []
        self.covered = np.zeros_like(self.density_map, dtype=bool)
        return self.covered.copy(), len(self.drone_positions)
    
    def compute_submodular_reward(self):
        return np.sum(self.density_map * self.covered)
    
    def step(self, action_indices):
        pos = self.possible_locations[action_indices]
        self.drone_positions.append(pos)
        x, y = pos
        for row_offset in range(-self.coverage_radius, self.coverage_radius + 1):
            for col_offset in range(-self.coverage_radius, self.coverage_radius + 1):
                neighbor_row = x + row_offset
                neighbor_col = y + col_offset
                if(0 <= neighbor_row < self.density_map.shape[0] and 0 <= neighbor_col < self.density_map.shape[1]):
                    self.covered[neighbor_row, neighbor_col] = True

        reward = self.compute_submodular_reward()
        done = len(self.drone_positions) >= self.n_drones
        return self.covered.copy(), reward, done
                    

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.screen.fill((255, 255, 255))

        for x in range(self.density_map.shape[0]):
            for y in range(self.density_map.shape[1]):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                red_intensity = int(255 * (self.density_map[x, y]))
                color = (red_intensity, 0, 0)
                if self.covered[x, y]:
                    color = (0, 200, 0)
                pygame.draw.rect(self.screen, color, rect)

        for pos in self.drone_positions:
            x, y = pos
            center = (y * self.cell_size + self.cell_size // 2, x * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, (0, 0, 255), center, self.cell_size // 3)
        pygame.display.flip()

    def close(self):
        pygame.quit()