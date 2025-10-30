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
        self.font = pygame.font.Font(None, 24)

    def reset(self):
        self.drone_positions = []
        self.covered = np.zeros_like(self.density_map, dtype=bool)
        self.visited_locations = set()
        return self.get_state()
    
    def get_state(self):
        return self.covered.copy().flatten()
    
    def compute_total_coverage(self):
        return np.sum(self.density_map * self.covered)
    
    def compute_marginal_gain(self, position):
        if position in self.visited_locations:
            return 0.0
        
        #calculate what would be newly covered
        x, y = position
        new_coverage = 0.0
        for row_offset in range(-self.coverage_radius, self.coverage_radius + 1):
            for col_offset in range(-self.coverage_radius, self.coverage_radius + 1):
                neighbor_row = x + row_offset
                neighbor_col = y + col_offset
                if(0 <= neighbor_row < self.density_map.shape[0] and 
                   0 <= neighbor_col < self.density_map.shape[1] and 
                   not self.covered[neighbor_row, neighbor_col]):
                    new_coverage += self.density_map[neighbor_row, neighbor_col]
            
        return new_coverage

    
    def step(self, action_indices):
        pos = self.possible_locations[action_indices]
        
        #calculate marginal gain BEFORE placing the drone
        marginal_gain = self.compute_marginal_gain(pos)

        #place drone and update coverage
        self.drone_positions.append(pos)
        self.visited_locations.add(pos)

        x, y = pos
        for row_offset in range(-self.coverage_radius, self.coverage_radius + 1):
            for col_offset in range(-self.coverage_radius, self.coverage_radius + 1):
                neighbor_row = x + row_offset
                neighbor_col = y + col_offset
                if(0 <= neighbor_row < self.density_map.shape[0] and 
                    0 <= neighbor_col < self.density_map.shape[1]):
                    self.covered[neighbor_row, neighbor_col] = True

        total_reward = self.compute_total_coverage()
        done = len(self.drone_positions) >= self.n_drones
        return self.get_state(), marginal_gain, total_reward, done
                    

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.screen.fill((255, 255, 255))

        for x in range(self.density_map.shape[0]):
            for y in range(self.density_map.shape[1]):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                density = self.density_map[x, y]
                if self.covered[x, y]:
                    color = (0, min(255, 100 + int(155 * density)), 0)
                else:
                    color = (min(255, 100 + int(155 * density)), 0, 0)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1) #grid lines

        for pos in self.drone_positions:
            x, y = pos
            center = (y * self.cell_size + self.cell_size // 2, x * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, (0, 0, 255), center, self.cell_size // 3)
            pygame.draw.circle(self.screen, (255, 255, 255), center, self.cell_size // 3, 2)

        # display info
        coverage = self.compute_total_coverage()
        info_text = f"Drones: {len(self.drone_positions)}/{self.n_drones} Coverage: {coverage:.1f}"
        text_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        pygame.display.flip()
        return True

    def close(self):
        pygame.quit()