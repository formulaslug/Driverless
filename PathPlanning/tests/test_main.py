"""Integration tests for path_planner.py"""
import numpy as np
import pytest
import time
import config as cfg
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from path_planner import plan_path
from track_generator import TrackGenerator
from utils import Mode, SimType
import yaml
import os

# Performance constraint from CLAUDE.md
TARGET_CYCLE_TIME_MS = 40  # 25 Hz requirement


# Create test tracks
def create_simple_straight_track():
    """ Straight track with 4 cone pairs """
    x_positions = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
    trackwidth = 3      # 3m track width
    
    cones, colors = [], []
    
    for x in x_positions:
        # Blue cones on the left
        cones.append([x, trackwidth/2])
        colors.append([0.95, 0.02, 0.02, 0.01])
        
        cones.append([x, -trackwidth/2])
        colors.append([0.02, 0.95, 0.02, 0.01])
        
    return np.array(cones), np.array(colors)

class TestOutputs:
    def setup_method(self):
        self.cones, self.colors = create_simple_straight_track()
        self.coordinate_confidence = np.zeros(len(self.cones))
        self.vehicle_pos = np.array([-2, 0])
        self.vehicle_heading = 0.0
        
    def test_plan_path_returns_tuple(self):
        """Verify plan_path returns a tuple."""

        result = plan_path(self.cones, self.coordinate_confidence, self.colors, self.vehicle_pos, self.vehicle_heading)

        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) == 2, "Result should have 2 elements"
        
    def test_output_shape(self):
        smooth_path, curvature = plan_path(self.cones, self.coordinate_confidence, self.colors, self.vehicle_pos, self.vehicle_heading)
        
        assert isinstance(smooth_path, np.ndarray)
        assert curvature.shape == (cfg.SPLINE_NUM_POINTS, )
        assert smooth_path.shape == (cfg.SPLINE_NUM_POINTS, 2)  # (100, 2)
        
    def test_dtype_is_float(self):
        smooth_path, curvature = plan_path(self.cones, self.coordinate_confidence, self.colors, self.vehicle_pos, self.vehicle_heading)
        
        assert np.issubdtype(smooth_path.dtype, np.floating)
        assert np.issubdtype(curvature.dtype, np.floating)
        
class TestEdgeCases:
    def test_empty_cones(self):
        cones = np.array([]).reshape(0, 2)
        colors = np.array([]).reshape(0, 4)
        coordinate_confidence = np.zeros(len(cones))
        res = plan_path(cones, coordinate_confidence, colors, np.array([0, 0]), 0.0)
        
        assert res == (None, None)
        
    def test_collinear_cones(self):                           
        cones = np.array([[0, 0], [2, 0], [4, 0], [6, 0]])    
        colors = np.array([[0.95, 0.02, 0.02, 0.01]] * 4)     
        coordinate_confidence = np.zeros(len(cones))
        result = plan_path(cones, coordinate_confidence, colors, np.array([-2, 0]), 0.0)                                                      
        assert result == (None, None)  
      
    def test_duplicate_cone_positions(self):                  
        """Two cones at same position."""                     
        cones = np.array([                                    
            [0, 1.5], [0, 1.5],  # Duplicate                
            [5, 1.5], [5, -1.5],                              
        ])                                                    
        colors = np.array([                                   
            [0.95, 0.02, 0.02, 0.01],                         
            [0.02, 0.95, 0.02, 0.01],                         
            [0.95, 0.02, 0.02, 0.01],                         
            [0.02, 0.95, 0.02, 0.01],                         
        ])                        
        
        coordinate_confidence = np.zeros(len(cones))                            
        # Should either return None or handle gracefully                                                  
        try:                                                  
            result = plan_path(cones, coordinate_confidence, colors, np.array([-2, 0]), 0.0)                                                 
        except Exception as e:                                
            pytest.fail(f"Should not crash: {e}")
            
    def test_all_blue_cones(self):                            
        cones, _ = create_simple_straight_track()    
        coordinate_confidence = np.zeros(len(cones))         
        colors = np.array([[0.95, 0.02, 0.02, 0.01]] * len(cones))                                               
        result = plan_path(cones, coordinate_confidence, colors, np.array([-2, 0]), 0.0)                                                      
        assert result == (None, None)                         
                                                                
    def test_all_yellow_cones(self):                          
        cones, _ = create_simple_straight_track()   
        coordinate_confidence = np.zeros(len(cones))          
        colors = np.array([[0.02, 0.95, 0.02, 0.01]] * len(cones))                                               
        result = plan_path(cones, coordinate_confidence, colors, np.array([-2, 0]), 0.0)                                                      
        assert result == (None, None)
        
    def test_vehicle_facing_backwards(self):                  
        cones, colors = create_simple_straight_track()     
        coordinate_confidence = np.zeros(len(cones))   
        result = plan_path(cones, coordinate_confidence, colors, np.array([-2, 0]), np.pi)  # Facing -x                                       
        assert result == (None, None)                             

class TestPerformance:
    def test_execution_time(self):
        import time
        cones, colors = create_simple_straight_track()
        coordinate_confidence = np.zeros(len(cones))
        start = time.perf_counter()
        path = plan_path(cones, coordinate_confidence, colors, np.array((-1, 0)), 0.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 40, f"Took {elapsed_ms:.2f}ms, limit is 40ms."
        
class TestVisualization:
    def setup_method(self):
        self.cones, self.colors = create_simple_straight_track()
        self.coordinate_confidence = np.zeros(len(self.cones))
        self.vehicle_pos = np.array([-2, 0])
        self.vehicle_heading = 0.0
        
    def test_visualize_straight_track(self):
        fig, ax = plt.subplots(figsize=(10,6))
        
        color_labels = np.argmax(self.colors, axis=1)
        blue_mask = color_labels == 0
        yellow_mask = color_labels == 1
        
        # Plot the cones
        ax.scatter(self.cones[blue_mask, 0], self.cones[blue_mask, 1], 
                   c='blue', s=80, label='Blue')
        ax.scatter(self.cones[yellow_mask, 0], self.cones[yellow_mask, 1], 
                   c='yellow', edgecolors='black', s=80, label='Yellow')
        
        # Plot the vehicle
        ax.scatter(*self.vehicle_pos, c='green', s=150, marker='s', label='Vehicle')
        ax.arrow(self.vehicle_pos[0], self.vehicle_pos[1],
                 1.5*np.cos(self.vehicle_heading), 1.5*np.sin(self.vehicle_heading),
                 head_width=0.3, color='green')
        
        smooth_path, _ = plan_path(self.cones, self.coordinate_confidence, self.colors, self.vehicle_pos, self.vehicle_heading)
        if smooth_path is not None:
            ax.plot(smooth_path[:, 0], smooth_path[:, 1],
                    c='red', linewidth=2, label='Path')
            
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Test Straight Track')
        plt.show()
    
    def test_visualize_generated_track(self):
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
        yaml_path = os.path.join(os.path.dirname(__file__), 'random_track.yaml')
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        # Access the coordinate data
        left = np.array(data['cones_left'])
        right = np.array(data['cones_right'])
    
        cones = np.concatenate((left,right), axis=0)
        coordinate_confidence = np.zeros(len(cones))
        colors = np.zeros((len(left) + len(right), 4))
        colors[:len(left), 0] = 0.9
        colors[len(left):, 1] = 0.9
        vehicle_pos = np.array([-2, 0])
        vehicle_heading = 0.0
        fig, ax = plt.subplots()
        
        # Plot the cones
        ax.scatter(left[:, 0], left[:, 1], 
                   c='blue', s=5)
        ax.scatter(right[:, 0], right[:, 1], 
                   c='yellow', edgecolors='grey', s=5)
        
        # Plot the vehicle
        ax.scatter(*vehicle_pos, c='green', s=15, marker='s', label='Vehicle')
        ax.arrow(vehicle_pos[0], vehicle_pos[1],
                 1.5*np.cos(vehicle_heading), 1.5*np.sin(vehicle_heading),
                 head_width=0.3, color='green')
        
        smooth_path, _ = plan_path(cones, coordinate_confidence, colors, vehicle_pos, vehicle_heading)
        if smooth_path is not None:
            ax.plot(smooth_path[2:, 0], smooth_path[2:, 1],
                    c='red', linewidth=2, label='Path')
        
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Track visualization')
        plt.show()
