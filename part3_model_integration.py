import numpy as np
import time
from simulation_and_control import feedback_lin_ctrl
from simulation_utilities import (
    SimulationSetup,
    RobotStateReader,
    SimulationExecutor,
    TargetReachingChecker
)
from common_utilities import ModelPersistence, ArrayOperations
from machine_learning_utilities import DatasetPreparator
from visualization_utilities import IntegrationTestVisualizer


class TrainedModelLoader:
    
    @staticmethod
    def load_model_from_file(model_filename):
        
        if model_filename.endswith('_mlp_model.pkl'):
            trained_model, feature_scaler, target_scaler = ModelPersistence.load_model_with_scalers(model_filename)
            return {
                'model': trained_model,
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'is_mlp_model': True
            }
        else:
            trained_model = ModelPersistence.load_model_only(model_filename)
            return {
                'model': trained_model,
                'feature_scaler': None,
                'target_scaler': None,
                'is_mlp_model': False
            }


class MLTrajectoryController:
    
    def __init__(self, model_package, number_of_joints):
        
        self.trained_model = model_package['model']
        self.feature_scaler = model_package['feature_scaler']
        self.target_scaler = model_package['target_scaler']
        self.is_mlp_model = model_package['is_mlp_model']
        self.number_of_joints = number_of_joints
    
    def predict_desired_joint_states(self, measured_joint_positions, target_cartesian_position):
        
        input_features = np.concatenate([measured_joint_positions, target_cartesian_position])
        input_features = input_features.reshape(1, -1)
        
        if self.is_mlp_model:
            input_features_normalized = self.feature_scaler.transform(input_features)
            predicted_output_normalized = self.trained_model.predict(input_features_normalized)
            predicted_output = self.target_scaler.inverse_transform(predicted_output_normalized)
        else:
            predicted_output = self.trained_model.predict(input_features)
        
        predicted_output = predicted_output.flatten()
        
        desired_joint_positions, desired_joint_velocities = DatasetPreparator.split_concatenated_output_into_positions_and_velocities(
            predicted_output,
            self.number_of_joints
        )
        
        return desired_joint_positions, desired_joint_velocities
    
    def clip_velocities_to_safety_limits(self, desired_joint_velocities, joint_velocity_limits):
        
        clipped_velocities = ArrayOperations.clip_values_to_limits(
            desired_joint_velocities,
            -np.array(joint_velocity_limits),
            np.array(joint_velocity_limits)
        )
        
        return clipped_velocities


class SingleTrajectoryExecutor:
    
    def __init__(self, simulation_setup, ml_controller, controlled_frame_name="panda_link8"):
        
        self.simulation_setup = simulation_setup
        self.ml_controller = ml_controller
        self.controlled_frame_name = controlled_frame_name
        self.low_level_proportional_gain = 1000
        self.low_level_derivative_gain = 100
        self.maximum_time_per_target = 15.0
        
    def execute_trajectory_to_target(self, target_cartesian_position, joint_velocity_limits):
        
        trajectory_data_recorder = {
            'measured_joint_positions': [],
            'measured_cartesian_positions': [],
            'desired_joint_positions': [],
            'desired_joint_velocities': [],
            'time_steps': []
        }
        
        current_simulation_time = 0
        target_start_time = 0
        
        target_cartesian_position_array = np.array(target_cartesian_position)
        
        while True:
            
            joint_measurements = RobotStateReader.read_joint_measurements(
                self.simulation_setup.simulation_interface
            )
            measured_joint_positions = joint_measurements['positions']
            measured_joint_velocities = joint_measurements['velocities']
            
            forward_kinematics_result = RobotStateReader.compute_forward_kinematics(
                self.simulation_setup.dynamic_model,
                measured_joint_positions,
                self.controlled_frame_name
            )
            current_cartesian_position = forward_kinematics_result['cartesian_position']
            
            desired_joint_positions, desired_joint_velocities = self.ml_controller.predict_desired_joint_states(
                measured_joint_positions,
                target_cartesian_position_array
            )
            
            desired_joint_velocities_clipped = self.ml_controller.clip_velocities_to_safety_limits(
                desired_joint_velocities,
                joint_velocity_limits
            )
            
            torque_command = feedback_lin_ctrl(
                self.simulation_setup.dynamic_model,
                measured_joint_positions,
                measured_joint_velocities,
                desired_joint_positions,
                desired_joint_velocities_clipped,
                self.low_level_proportional_gain,
                self.low_level_derivative_gain
            )
            
            SimulationExecutor.execute_single_control_step(
                self.simulation_setup.simulation_interface,
                self.simulation_setup.motor_command,
                torque_command,
                self.simulation_setup.number_of_joints
            )
            
            SimulationExecutor.update_visualization_display(
                self.simulation_setup.simulation_interface,
                self.simulation_setup.dynamic_model
            )
            
            if SimulationExecutor.check_for_quit_keyboard_event(self.simulation_setup.simulation_interface):
                return None
            
            trajectory_data_recorder['measured_joint_positions'].append(measured_joint_positions.copy())
            trajectory_data_recorder['measured_cartesian_positions'].append(current_cartesian_position.copy())
            trajectory_data_recorder['desired_joint_positions'].append(desired_joint_positions.copy())
            trajectory_data_recorder['desired_joint_velocities'].append(desired_joint_velocities_clipped.copy())
            trajectory_data_recorder['time_steps'].append(current_simulation_time)
            
            time.sleep(0.01)
            current_simulation_time += self.simulation_setup.simulation_time_step
            
            target_reached, position_error = TargetReachingChecker.check_if_target_reached(
                current_cartesian_position,
                target_cartesian_position_array
            )
            
            if target_reached:
                print(f"Target reached! Final position error: {position_error:.6f}")
                break
            
            timeout_exceeded, elapsed_time = TargetReachingChecker.check_if_timeout_exceeded(
                current_simulation_time,
                target_start_time,
                self.maximum_time_per_target
            )
            
            if timeout_exceeded:
                print(f"Timeout exceeded. Final position error: {position_error:.6f}")
                break
        
        final_position_error = ArrayOperations.compute_euclidean_distance(
            target_cartesian_position_array,
            current_cartesian_position
        )
        
        trajectory_data = {
            'target_position': target_cartesian_position_array,
            'measured_joint_positions': np.array(trajectory_data_recorder['measured_joint_positions']),
            'measured_cartesian_positions': np.array(trajectory_data_recorder['measured_cartesian_positions']),
            'desired_joint_positions': np.array(trajectory_data_recorder['desired_joint_positions']),
            'desired_joint_velocities': np.array(trajectory_data_recorder['desired_joint_velocities']),
            'time_steps': np.array(trajectory_data_recorder['time_steps']),
            'final_error': final_position_error
        }
        
        return trajectory_data


class ModelIntegrationPipeline:
    
    def __init__(self, model_filename='part2_trajectory_prediction_mlp_model.pkl'):
        
        self.model_filename = model_filename
        self.model_package = None
        self.simulation_setup = None
        self.ml_controller = None
        self.trajectory_executor = None
        self.test_target_positions = [
            [0.5, 0.2, 0.4],
            [0.4, -0.1, 0.5]
        ]
        self.all_trajectories_data = []
        
    def load_trained_model(self):
        
        print(f"Loading trained model from {self.model_filename}...")
        self.model_package = TrainedModelLoader.load_model_from_file(self.model_filename)
        
        model_type = 'MLP' if self.model_package['is_mlp_model'] else 'Random Forest'
        print(f"Model type: {model_type}")
        
        return self
    
    def initialize_simulation_environment(self):
        
        self.simulation_setup = SimulationSetup()
        self.simulation_setup.initialize_simulation_environment()
        
        robot_state = self.simulation_setup.get_initial_robot_state()
        print(f"Initial joint angles: {robot_state['initial_joint_angles']}")
        
        return self
    
    def create_ml_trajectory_controller(self):
        
        self.ml_controller = MLTrajectoryController(
            self.model_package,
            self.simulation_setup.number_of_joints
        )
        
        self.trajectory_executor = SingleTrajectoryExecutor(
            self.simulation_setup,
            self.ml_controller
        )
        
        return self
    
    def execute_all_test_trajectories(self):
        
        robot_state = self.simulation_setup.get_initial_robot_state()
        joint_velocity_limits = robot_state['joint_velocity_limits']
        
        for target_index, target_position in enumerate(self.test_target_positions):
            
            print(f"\n{'=' * 60}")
            print(f"Test Trajectory {target_index + 1}/{len(self.test_target_positions)}")
            print(f"Target position: {target_position}")
            print(f"{'=' * 60}")
            
            self.simulation_setup.reset_simulation_to_initial_state()
            self.simulation_setup.initialize_simulation_environment()
            
            self.trajectory_executor.simulation_setup = self.simulation_setup
            
            trajectory_data = self.trajectory_executor.execute_trajectory_to_target(
                target_position,
                joint_velocity_limits
            )
            
            if trajectory_data is None:
                print("Trajectory execution interrupted by user.")
                break
            
            self.print_trajectory_summary(trajectory_data)
            
            self.all_trajectories_data.append(trajectory_data)
        
        return self
    
    def print_trajectory_summary(self, trajectory_data):
        
        print(f"\nTrajectory Summary:")
        print(f"  Desired target: {trajectory_data['target_position']}")
        print(f"  Final position: {trajectory_data['measured_cartesian_positions'][-1]}")
        print(f"  Final error: {trajectory_data['final_error']:.6f}")
        print(f"  Time elapsed: {trajectory_data['time_steps'][-1]:.2f} seconds")
    
    def generate_trajectory_visualizations(self):
        
        print(f"\n{'=' * 60}")
        print("Generating trajectory visualizations...")
        print(f"{'=' * 60}")
        
        IntegrationTestVisualizer.visualize_all_test_trajectories(
            self.all_trajectories_data,
            number_of_joints=self.simulation_setup.number_of_joints,
            save_prefix='part3'
        )
        
        return self


def main():
    
    print("Part 3: Model Integration in Simulation")
    print("=" * 60)
    
    integration_pipeline = ModelIntegrationPipeline(
        model_filename='part2_trajectory_prediction_mlp_model.pkl'
    )
    
    integration_pipeline.load_trained_model()
    
    integration_pipeline.initialize_simulation_environment()
    
    integration_pipeline.create_ml_trajectory_controller()
    
    integration_pipeline.execute_all_test_trajectories()
    
    integration_pipeline.generate_trajectory_visualizations()
    
    print(f"\n{'=' * 60}")
    print("Part 3 completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
