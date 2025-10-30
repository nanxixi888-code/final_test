import numpy as np
import os
from simulation_and_control import pb, MotorCommands, PinWrapper


class SimulationSetup:
    
    def __init__(self, configuration_file_name="pandaconfig.json"):
        
        self.configuration_file_name = configuration_file_name
        self.root_directory = os.path.dirname(os.path.abspath(__file__))
        self.simulation_interface = None
        self.dynamic_model = None
        self.number_of_joints = None
        self.motor_command = None
        self.simulation_time_step = None
        
    def initialize_simulation_environment(self):
        
        self.simulation_interface = pb.SimInterface(
            self.configuration_file_name, 
            conf_file_path_ext=self.root_directory
        )
        
        active_joint_names = self.simulation_interface.getNameActiveJoints()
        active_joint_names = np.expand_dims(np.array(active_joint_names), axis=0)
        
        source_names = ["pybullet"]
        
        self.dynamic_model = PinWrapper(
            self.configuration_file_name, 
            "pybullet", 
            active_joint_names, 
            source_names, 
            False, 
            0, 
            self.root_directory
        )
        
        self.number_of_joints = self.dynamic_model.getNumberofActuatedJoints()
        self.motor_command = MotorCommands()
        self.simulation_time_step = self.simulation_interface.GetTimeStep()
        
        return self
    
    def get_initial_robot_state(self):
        
        initial_joint_angles = self.simulation_interface.GetInitMotorAngles()
        lower_joint_limits, upper_joint_limits = self.simulation_interface.GetBotJointsLimit()
        joint_velocity_limits = self.simulation_interface.GetBotJointsVelLimit()
        
        return {
            'initial_joint_angles': initial_joint_angles,
            'lower_joint_limits': lower_joint_limits,
            'upper_joint_limits': upper_joint_limits,
            'joint_velocity_limits': joint_velocity_limits
        }
    
    def reset_simulation_to_initial_state(self):
        
        self.simulation_interface.GetPyBulletClient().resetSimulation()
        self.simulation_interface = pb.SimInterface(
            self.configuration_file_name, 
            conf_file_path_ext=self.root_directory
        )


class RobotStateReader:
    
    @staticmethod
    def read_joint_measurements(simulation_interface, robot_index=0):
        
        measured_joint_positions = simulation_interface.GetMotorAngles(robot_index)
        measured_joint_velocities = simulation_interface.GetMotorVelocities(robot_index)
        measured_joint_accelerations = simulation_interface.ComputeMotorAccelerationTMinusOne(robot_index)
        
        return {
            'positions': measured_joint_positions,
            'velocities': measured_joint_velocities,
            'accelerations': measured_joint_accelerations
        }
    
    @staticmethod
    def compute_forward_kinematics(dynamic_model, joint_positions, controlled_frame_name):
        
        cartesian_position, rotation_matrix = dynamic_model.ComputeFK(joint_positions, controlled_frame_name)
        cartesian_orientation = dynamic_model.GetQuaternionFromRotationMatrix(rotation_matrix)
        
        return {
            'cartesian_position': cartesian_position,
            'rotation_matrix': rotation_matrix,
            'cartesian_orientation': cartesian_orientation
        }


class ControlCommandGenerator:
    
    @staticmethod
    def compute_cartesian_velocity_from_position_error(current_position, target_position, proportional_gain):
        
        position_error = target_position - current_position
        desired_velocity = position_error * proportional_gain
        return desired_velocity
    
    @staticmethod
    def clip_joint_velocities_to_safety_limits(desired_velocities, velocity_limits):
        
        velocity_limits_array = np.array(velocity_limits)
        clipped_velocities = np.clip(desired_velocities, -velocity_limits_array, velocity_limits_array)
        return clipped_velocities


class SimulationExecutor:
    
    @staticmethod
    def execute_single_control_step(simulation_interface, motor_command, torque_command, number_of_joints):
        
        motor_command.SetControlCmd(torque_command, ["torque"] * number_of_joints)
        simulation_interface.Step(motor_command, "torque")
    
    @staticmethod
    def update_visualization_display(simulation_interface, dynamic_model):
        
        if dynamic_model.visualizer:
            for robot_index in range(len(simulation_interface.bot)):
                current_joint_angles = simulation_interface.GetMotorAngles(robot_index)
                dynamic_model.DisplayModel(current_joint_angles)
    
    @staticmethod
    def check_for_quit_keyboard_event(simulation_interface):
        
        keyboard_events = simulation_interface.GetPyBulletClient().getKeyboardEvents()
        quit_key_code = ord('q')
        
        if quit_key_code in keyboard_events and keyboard_events[quit_key_code]:
            if simulation_interface.GetPyBulletClient().KEY_WAS_TRIGGERED:
                return True
        
        return False


class TrajectoryDataRecorder:
    
    def __init__(self):
        
        self.measured_joint_positions_over_time = []
        self.measured_joint_velocities_over_time = []
        self.measured_cartesian_positions_over_time = []
        self.measured_cartesian_orientations_over_time = []
        self.desired_joint_positions_over_time = []
        self.desired_joint_velocities_over_time = []
        self.torque_commands_over_time = []
        self.final_desired_cartesian_positions_over_time = []
        self.final_desired_cartesian_orientations_over_time = []
    
    def record_single_timestep_data(self, measurement_dict):
        
        self.measured_joint_positions_over_time.append(measurement_dict['measured_joint_positions'].copy())
        self.measured_joint_velocities_over_time.append(measurement_dict['measured_joint_velocities'].copy())
        self.measured_cartesian_positions_over_time.append(measurement_dict['measured_cartesian_position'].copy())
        self.measured_cartesian_orientations_over_time.append(measurement_dict['measured_cartesian_orientation'].copy())
        self.desired_joint_positions_over_time.append(measurement_dict['desired_joint_positions'].copy())
        self.desired_joint_velocities_over_time.append(measurement_dict['desired_joint_velocities'].copy())
        self.torque_commands_over_time.append(measurement_dict['torque_command'].copy())
        self.final_desired_cartesian_positions_over_time.append(measurement_dict['final_target_cartesian_position'].copy())
        self.final_desired_cartesian_orientations_over_time.append(measurement_dict['final_target_cartesian_orientation'].copy())
    
    def convert_recorded_data_to_numpy_arrays(self):
        
        return {
            'q_mes': np.array(self.measured_joint_positions_over_time),
            'qd_mes': np.array(self.measured_joint_velocities_over_time),
            'cart_pos': np.array(self.measured_cartesian_positions_over_time),
            'cart_ori': np.array(self.measured_cartesian_orientations_over_time),
            'q_des': np.array(self.desired_joint_positions_over_time),
            'qd_des': np.array(self.desired_joint_velocities_over_time),
            'tau_cmd': np.array(self.torque_commands_over_time),
            'desired_cartesian_pos': np.array(self.final_desired_cartesian_positions_over_time),
            'desired_cartesian_ori': np.array(self.final_desired_cartesian_orientations_over_time)
        }
    
    def get_number_of_recorded_timesteps(self):
        
        return len(self.measured_joint_positions_over_time)


class TargetReachingChecker:
    
    @staticmethod
    def check_if_target_reached(current_position, target_position, threshold=0.01):
        
        position_error = np.linalg.norm(target_position - current_position)
        return position_error < threshold, position_error
    
    @staticmethod
    def check_if_timeout_exceeded(current_time, start_time, maximum_duration):
        
        elapsed_time = current_time - start_time
        return elapsed_time >= maximum_duration, elapsed_time

