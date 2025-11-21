import numpy as np

class FloodCoverageEnvironment:
    def __init__(self, density_map, n_drones, coverage_radius):
        self.density_map = np.array(density_map)
        self.n_drones = n_drones
        self.coverage_radius = coverage_radius
        self.possible_locations = list(np.ndindex(self.density_map.shape))
        self.reset()
        
    def reset(self):
        self.drone_positions = []
        self.covered = np.zeros_like(self.density_map, dtype=bool)
        self.visited_locations = set()
        return self.get_state()
    
    def get_state(self):
        coverage_layer = self.covered.astype(np.float32)
        density_layer = self.density_map
        state = np.stack([coverage_layer, density_layer], axis=0)
        return state.flatten()
    
    def compute_total_coverage(self):
        return np.sum(self.density_map * self.covered)
    
    # F(s|τ_{0:j}) = F(τ_{0:j} ∪ {s}) - F(τ_{0:j})
    # F is the coverage function
    # τ_{0:j} is the trajectory (sequence of positions) up to time j
    # s is the new state we are considering
    # F(s|τ_{0:j}) means how much NEW coverage would be gained by adding state s
    #   given that all the previous states in F(τ_{0:j})
    def compute_marginal_gain(self, position):
        if position in self.visited_locations:
            return 0.0 #diminishing returns: already visited locations provide zero marginal gain
        
        #calculate what would be newly covered
        x, y = position
        new_coverage = 0.0
        for row_offset in range(-self.coverage_radius, self.coverage_radius + 1):
            for col_offset in range(-self.coverage_radius, self.coverage_radius + 1):
                if abs(row_offset) + abs(col_offset) > self.coverage_radius:
                    continue # skip if outside coverage radius (Manhattan distance)

                neighbor_row = x + row_offset
                neighbor_col = y + col_offset
                if(0 <= neighbor_row < self.density_map.shape[0] and 
                   0 <= neighbor_col < self.density_map.shape[1] and 
                   not self.covered[neighbor_row, neighbor_col]):
                    new_coverage += self.density_map[neighbor_row, neighbor_col] #weight by importance/density
            
        return new_coverage

    
    def step(self, action_indices):
        pos = self.possible_locations[action_indices]
        
        #normal RL: computes gain after updating state
        #submodular rl: calculate marginal gain BEFORE state transition(placing the drone)
        #tells how much this action actually given previous actions
        marginal_gain = self.compute_marginal_gain(pos)

        #place drone and update coverage (state transition)
        self.drone_positions.append(pos)
        self.visited_locations.add(pos) # history tracking

        x, y = pos
        for row_offset in range(-self.coverage_radius, self.coverage_radius + 1):
            for col_offset in range(-self.coverage_radius, self.coverage_radius + 1):
                if abs(row_offset) + abs(col_offset) > self.coverage_radius:
                    continue

                neighbor_row = x + row_offset
                neighbor_col = y + col_offset
                if(0 <= neighbor_row < self.density_map.shape[0] and 
                    0 <= neighbor_col < self.density_map.shape[1]):
                    self.covered[neighbor_row, neighbor_col] = True

        total_reward = self.compute_total_coverage()
        done = len(self.drone_positions) >= self.n_drones
        return self.get_state(), marginal_gain, total_reward, done
                    
    def get_env_info(self):
        return {
            'density_map': self.density_map,
            'drone_positions': self.drone_positions,
            'covered': self.covered,
            'n_drones': self.n_drones,
            'coverage_radius': self.coverage_radius
        }