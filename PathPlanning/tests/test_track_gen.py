from track_generator import TrackGenerator
from utils import Mode, SimType
import yaml
import matplotlib.pyplot as plt
import numpy as np

def visualize_track(left, right):
        fig, ax = plt.subplots()
        
        left_x = left[:, 0]
        left_y = left[:, 1]
        right_x = right[:, 0]
        right_y = right[:, 1]
        # Plot the cones
        ax.scatter(left_x, left_y, 
                   c='blue', s=5)
        ax.scatter(right_x, right_y, 
                   c='yellow', edgecolors='grey', s=5)
        
        # Plot the vehicle
        # ax.scatter(*self.vehicle_pos, c='green', s=150, marker='s', label='Vehicle')
        # ax.arrow(self.vehicle_pos[0], self.vehicle_pos[1],
        #          1.5*np.cos(self.vehicle_heading), 1.5*np.sin(self.vehicle_heading),
        #          head_width=0.3, color='green')
        
        # smooth_path, _ = plan_path(self.cones, self.colors, self.vehicle_pos, self.vehicle_heading)
        # if smooth_path is not None:
        #     ax.plot(smooth_path[:, 0], smooth_path[:, 1],
        #             c='red', linewidth=2, label='Path')
        
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Track visualization')
        plt.show()
        
if __name__ == "__main__":
    
    # Configure track generation
    track_gen = TrackGenerator(
        n_points=60,           # Voronoi points
        n_regions=20,          # Regions to select
        min_bound=0.,          # Minimum x/y bound
        max_bound=150.,        # Maximum x/y bound
        mode=Mode.EXTEND,      # Selection mode
        plot_track=False,
        visualise_voronoi=False,
        create_output_file=True,
        output_location='/',
        sim_type=SimType.FSSIM
    )

    track_gen.create_track()

    # Open and read the YAML file
    with open('random_track.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Access the coordinate data
    cones_left = np.array(data['cones_left'])
    cones_right = np.array(data['cones_right'])
    
    visualize_track(cones_left, cones_right)