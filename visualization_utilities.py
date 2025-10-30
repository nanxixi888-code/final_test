import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class RegressionVisualizationPlotter:
    
    @staticmethod
    def plot_predicted_vs_actual_for_single_output(predicted_values, actual_values, 
                                                    output_name, dataset_name):
        
        plt.scatter(actual_values, predicted_values, alpha=0.5)
        plt.plot([actual_values.min(), actual_values.max()],
                 [actual_values.min(), actual_values.max()],
                 'r--', lw=2)
        plt.xlabel(f'Actual {output_name}')
        plt.ylabel(f'Predicted {output_name}')
        plt.title(f'{dataset_name}: Predicted vs Actual {output_name}')
        plt.grid(True)
    
    @staticmethod
    def plot_residual_histogram_for_single_output(predicted_values, actual_values, 
                                                   output_name, dataset_name):
        
        residuals = actual_values - predicted_values
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residual Error')
        plt.ylabel('Frequency')
        plt.title(f'{dataset_name}: {output_name} Residual Distribution')
        plt.grid(True)
    
    @staticmethod
    def create_dual_plot_for_single_joint(predicted_values, actual_values, 
                                          joint_index, dataset_name, output_type='Torque'):
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        RegressionVisualizationPlotter.plot_predicted_vs_actual_for_single_output(
            predicted_values[:, joint_index],
            actual_values[:, joint_index],
            f'{output_type} - Joint {joint_index + 1}',
            dataset_name
        )
        
        plt.subplot(1, 2, 2)
        RegressionVisualizationPlotter.plot_residual_histogram_for_single_output(
            predicted_values[:, joint_index],
            actual_values[:, joint_index],
            f'{output_type} - Joint {joint_index + 1}',
            dataset_name
        )
        
        plt.tight_layout()


class TorquePredictionVisualizer:
    
    @staticmethod
    def plot_all_joints_predictions(predicted_torques, actual_torques, 
                                    dataset_name, number_of_joints=7, save_prefix=''):
        
        for joint_index in range(number_of_joints):
            RegressionVisualizationPlotter.create_dual_plot_for_single_joint(
                predicted_torques,
                actual_torques,
                joint_index,
                dataset_name,
                output_type='Torque Command'
            )
            
            if save_prefix:
                filename = f'{save_prefix}_{dataset_name.lower()}_joint_{joint_index + 1}.png'
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()


class TrajectoryPredictionVisualizer:
    
    @staticmethod
    def plot_model_comparison_for_single_joint(predicted_mlp, predicted_rf, actual_values,
                                               joint_index, number_of_joints):
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.scatter(actual_values[:, joint_index], predicted_mlp[:, joint_index], 
                   alpha=0.5, label='MLP')
        plt.scatter(actual_values[:, joint_index], predicted_rf[:, joint_index], 
                   alpha=0.5, label='Random Forest')
        plt.plot([actual_values[:, joint_index].min(), actual_values[:, joint_index].max()],
                 [actual_values[:, joint_index].min(), actual_values[:, joint_index].max()],
                 'r--', lw=2)
        plt.xlabel('Actual Desired Joint Position')
        plt.ylabel('Predicted Desired Joint Position')
        plt.title(f'Joint Position {joint_index + 1}: Predicted vs Actual')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        residuals_mlp = actual_values[:, joint_index] - predicted_mlp[:, joint_index]
        residuals_rf = actual_values[:, joint_index] - predicted_rf[:, joint_index]
        plt.hist(residuals_mlp, bins=50, alpha=0.5, label='MLP')
        plt.hist(residuals_rf, bins=50, alpha=0.5, label='Random Forest')
        plt.xlabel('Residual Error')
        plt.ylabel('Frequency')
        plt.title(f'Joint Position {joint_index + 1}: Residual Distribution')
        plt.legend()
        plt.grid(True)
        
        velocity_index = joint_index + number_of_joints
        
        plt.subplot(2, 2, 3)
        plt.scatter(actual_values[:, velocity_index], predicted_mlp[:, velocity_index], 
                   alpha=0.5, label='MLP')
        plt.scatter(actual_values[:, velocity_index], predicted_rf[:, velocity_index], 
                   alpha=0.5, label='Random Forest')
        plt.plot([actual_values[:, velocity_index].min(), actual_values[:, velocity_index].max()],
                 [actual_values[:, velocity_index].min(), actual_values[:, velocity_index].max()],
                 'r--', lw=2)
        plt.xlabel('Actual Desired Joint Velocity')
        plt.ylabel('Predicted Desired Joint Velocity')
        plt.title(f'Joint Velocity {joint_index + 1}: Predicted vs Actual')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        residuals_mlp_vel = actual_values[:, velocity_index] - predicted_mlp[:, velocity_index]
        residuals_rf_vel = actual_values[:, velocity_index] - predicted_rf[:, velocity_index]
        plt.hist(residuals_mlp_vel, bins=50, alpha=0.5, label='MLP')
        plt.hist(residuals_rf_vel, bins=50, alpha=0.5, label='Random Forest')
        plt.xlabel('Residual Error')
        plt.ylabel('Frequency')
        plt.title(f'Joint Velocity {joint_index + 1}: Residual Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
    
    @staticmethod
    def plot_comparison_for_all_joints(predicted_mlp, predicted_rf, actual_values,
                                      number_of_joints=7, save_prefix=''):
        
        for joint_index in range(number_of_joints):
            TrajectoryPredictionVisualizer.plot_model_comparison_for_single_joint(
                predicted_mlp,
                predicted_rf,
                actual_values,
                joint_index,
                number_of_joints
            )
            
            if save_prefix:
                filename = f'{save_prefix}_comparison_joint_{joint_index + 1}.png'
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()


class TrajectoryVisualizationPlotter:
    
    @staticmethod
    def plot_2d_trajectory_projection(cartesian_positions, target_position, 
                                     axis_indices, axis_labels, subplot_position, title):
        
        plt.subplot(*subplot_position)
        plt.plot(cartesian_positions[:, axis_indices[0]], 
                cartesian_positions[:, axis_indices[1]], 
                'b-', linewidth=2, label='Trajectory')
        plt.plot(cartesian_positions[0, axis_indices[0]], 
                cartesian_positions[0, axis_indices[1]], 
                'go', markersize=10, label='Start')
        plt.plot(cartesian_positions[-1, axis_indices[0]], 
                cartesian_positions[-1, axis_indices[1]], 
                'ro', markersize=10, label='End')
        plt.plot(target_position[axis_indices[0]], 
                target_position[axis_indices[1]], 
                'r*', markersize=15, label='Target')
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
    @staticmethod
    def plot_3d_trajectory_view(cartesian_positions, target_position, subplot_position, title):
        
        ax = plt.subplot(*subplot_position, projection='3d')
        ax.plot3D(cartesian_positions[:, 0], cartesian_positions[:, 1], 
                 cartesian_positions[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax.scatter(cartesian_positions[0, 0], cartesian_positions[0, 1], 
                  cartesian_positions[0, 2], c='g', marker='o', s=100, label='Start')
        ax.scatter(cartesian_positions[-1, 0], cartesian_positions[-1, 1], 
                  cartesian_positions[-1, 2], c='r', marker='o', s=100, label='End')
        ax.scatter(target_position[0], target_position[1], target_position[2], 
                  c='r', marker='*', s=200, label='Target')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(title)
        ax.legend()
    
    @staticmethod
    def plot_distance_to_target_over_time(time_steps, cartesian_positions, 
                                         target_position, subplot_position, title):
        
        plt.subplot(*subplot_position)
        distances_to_target = np.linalg.norm(cartesian_positions - target_position, axis=1)
        plt.plot(time_steps, distances_to_target, 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance to Target (m)')
        plt.title(title)
        plt.grid(True)
    
    @staticmethod
    def create_comprehensive_trajectory_plot(trajectory_data, trajectory_index):
        
        plt.figure(figsize=(15, 10))
        
        cartesian_positions = trajectory_data['measured_cartesian_positions']
        target_position = trajectory_data['target_position']
        time_steps = trajectory_data['time_steps']
        
        TrajectoryVisualizationPlotter.plot_2d_trajectory_projection(
            cartesian_positions,
            target_position,
            [0, 1],
            ['X Position (m)', 'Y Position (m)'],
            (2, 2, 1),
            f'Trajectory {trajectory_index + 1}: XY Plane'
        )
        
        TrajectoryVisualizationPlotter.plot_2d_trajectory_projection(
            cartesian_positions,
            target_position,
            [0, 2],
            ['X Position (m)', 'Z Position (m)'],
            (2, 2, 2),
            f'Trajectory {trajectory_index + 1}: XZ Plane'
        )
        
        TrajectoryVisualizationPlotter.plot_3d_trajectory_view(
            cartesian_positions,
            target_position,
            (2, 2, 3),
            f'Trajectory {trajectory_index + 1}: 3D View'
        )
        
        TrajectoryVisualizationPlotter.plot_distance_to_target_over_time(
            time_steps,
            cartesian_positions,
            target_position,
            (2, 2, 4),
            f'Trajectory {trajectory_index + 1}: Distance to Target Over Time'
        )
        
        plt.tight_layout()
    
    @staticmethod
    def create_joint_tracking_plot(trajectory_data, trajectory_index, number_of_joints=7):
        
        plt.figure(figsize=(15, 10))
        
        measured_positions = trajectory_data['measured_joint_positions']
        desired_positions = trajectory_data['desired_joint_positions']
        time_steps = trajectory_data['time_steps']
        
        for joint_index in range(number_of_joints):
            plt.subplot(3, 3, joint_index + 1)
            plt.plot(time_steps, measured_positions[:, joint_index], 
                    'b-', linewidth=2, label='Measured')
            plt.plot(time_steps, desired_positions[:, joint_index], 
                    'r--', linewidth=2, label='Desired')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (rad)')
            plt.title(f'Joint {joint_index + 1}')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()


class IntegrationTestVisualizer:
    
    @staticmethod
    def visualize_all_test_trajectories(all_trajectories_data, number_of_joints=7, save_prefix=''):
        
        for trajectory_index, trajectory_data in enumerate(all_trajectories_data):
            
            TrajectoryVisualizationPlotter.create_comprehensive_trajectory_plot(
                trajectory_data,
                trajectory_index
            )
            
            if save_prefix:
                filename = f'{save_prefix}_trajectory_{trajectory_index + 1}_visualization.png'
                plt.savefig(filename, dpi=150)
                print(f"Saved visualization for trajectory {trajectory_index + 1}")
                plt.close()
            else:
                plt.show()
            
            TrajectoryVisualizationPlotter.create_joint_tracking_plot(
                trajectory_data,
                trajectory_index,
                number_of_joints
            )
            
            if save_prefix:
                filename = f'{save_prefix}_trajectory_{trajectory_index + 1}_joint_tracking.png'
                plt.savefig(filename, dpi=150)
                print(f"Saved joint tracking visualization for trajectory {trajectory_index + 1}")
                plt.close()
            else:
                plt.show()

