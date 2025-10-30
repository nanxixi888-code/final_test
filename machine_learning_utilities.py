# machine_learning_utilities.py

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# --- 新增 Imports ---
from typing import List
# 导入教授的 Rollout 类，用于类型提示
try:
    from rollout_loader import Rollout
except ImportError:
    print("Warning: rollout_loader.py not found. Mocking Rollout class.")
    from dataclasses import dataclass
    @dataclass
    class Rollout:
        # 这是我们为 Part 1 & 2 修改后的完整数据结构
        q_mes_all: List[List[float]]
        qd_mes_all: List[List[float]]
        tau_mes_all: List[List[float]]
        q_d_all: List[List[float]]
        qd_d_all: List[List[float]]
        final_target_pos_all: List[List[float]]


class DatasetPreparator:
    
    # --- Part 1 函数 (最终版) ---
    @staticmethod
    def extract_torque_prediction_features_and_targets(loaded_rollouts: List[Rollout]):
        
        all_input_features = []
        all_output_targets = []

        for rollout in loaded_rollouts:
            
            # --- 提取 Part 1 所需数据 ---
            # 感谢我们修复了 data_generator, 我们现在有 q_d_all (q_des) 了!
            q_mes = np.array(rollout.q_mes_all)    
            q_des = np.array(rollout.q_d_all)      
            tau_mes = np.array(rollout.tau_mes_all) 

            # --- 定义 Part 1 特征 (符合 PDF 要求) ---
            # 输入: "current angle error" (q_des - q_mes) [cite: 55-56]
            input_features = q_des - q_mes
            
            # 输出: "Torque commands" [cite: 59]
            output_targets = tau_mes                    
            
            all_input_features.append(input_features)
            all_output_targets.append(output_targets)
        
        if not all_input_features:
            print("错误：(Part 1) 没有从 rollouts 中提取到任何数据。")
            return None, None
            
        final_input_features = np.vstack(all_input_features)
        final_output_targets = np.vstack(all_output_targets)
        
        return final_input_features, final_output_targets
    
    # --- Part 2 函数 (最终版) ---
    @staticmethod
    def extract_trajectory_prediction_features_and_targets(loaded_rollouts: List[Rollout]):
        
        all_input_features = []
        all_output_targets = []

        for rollout in loaded_rollouts:
            
            # --- 提取 Part 2 所需数据 ---
            # (这是我们修复 data_generator 后才有的数据)
            
            # Part 2 输入 [cite: 70, 73]
            measured_joint_positions = np.array(rollout.q_mes_all)
            final_target_cartesian_positions = np.array(rollout.final_target_pos_all) 
            
            # Part 2 输出 [cite: 72]
            desired_joint_positions = np.array(rollout.q_d_all)
            desired_joint_velocities = np.array(rollout.qd_d_all)
            
            # --- 定义 Part 2 特征 (符合 PDF 要求) ---
            # (N, 7) + (N, 3) -> (N, 10)
            input_features = np.concatenate([measured_joint_positions, final_target_cartesian_positions], axis=1)
            
            # (N, 7) + (N, 7) -> (N, 14)
            output_targets = np.concatenate([desired_joint_positions, desired_joint_velocities], axis=1)
            
            all_input_features.append(input_features)
            all_output_targets.append(output_targets)
            
        if not all_input_features:
            print("错误：(Part 2) 没有从 rollouts 中提取到任何数据。")
            return None, None
            
        final_input_features = np.vstack(all_input_features)
        final_output_targets = np.vstack(all_output_targets)
        
        return final_input_features, final_output_targets
    
    @staticmethod
    def split_concatenated_output_into_positions_and_velocities(concatenated_output, number_of_joints):
        
        # ... (这个函数保持原样, 它是完美的) ...
        desired_joint_positions = concatenated_output[:, :number_of_joints]
        desired_joint_velocities = concatenated_output[:, number_of_joints:]
        
        return desired_joint_positions, desired_joint_velocities


# --- 你其他的类 (MLPRegressorTrainer, RandomForestRegressorTrainer, ModelPredictor) ---
# --- 保持原样，它们不需要修改 ---

class MLPRegressorTrainer:
    
    @staticmethod
    def get_default_hyperparameter_grid():
        
        hyperparameter_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [1000]
        }
        
        return hyperparameter_grid
    
    @staticmethod
    def create_mlp_regressor_base_model(random_seed=42, enable_early_stopping=True, validation_fraction=0.1):
        
        multilayer_perceptron_model = MLPRegressor(
            random_state=random_seed,
            early_stopping=enable_early_stopping,
            validation_fraction=validation_fraction
        )
        
        return multilayer_perceptron_model
    
    @staticmethod
    def train_with_grid_search_cross_validation(input_features_train, output_targets_train, 
                                                hyperparameter_grid=None, cross_validation_folds=3):
        
        if hyperparameter_grid is None:
            hyperparameter_grid = MLPRegressorTrainer.get_default_hyperparameter_grid()
        
        base_model = MLPRegressorTrainer.create_mlp_regressor_base_model()
        
        grid_search_cross_validator = GridSearchCV(
            base_model,
            hyperparameter_grid,
            cv=cross_validation_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search_cross_validator.fit(input_features_train, output_targets_train)
        
        print(f"Best MLP hyperparameters found: {grid_search_cross_validator.best_params_}")
        
        best_trained_model = grid_search_cross_validator.best_estimator_
        
        return best_trained_model


class RandomForestRegressorTrainer:
    
    @staticmethod
    def get_default_hyperparameter_grid():
        
        hyperparameter_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        return hyperparameter_grid
    
    @staticmethod
    def create_random_forest_regressor_base_model(random_seed=42):
        
        random_forest_model = RandomForestRegressor(
            random_state=random_seed,
            n_jobs=-1
        )
        
        return random_forest_model
    
    @staticmethod
    def train_with_grid_search_cross_validation(input_features_train, output_targets_train,
                                                hyperparameter_grid=None, cross_validation_folds=3):
        
        if hyperparameter_grid is None:
            hyperparameter_grid = RandomForestRegressorTrainer.get_default_hyperparameter_grid()
        
        base_model = RandomForestRegressorTrainer.create_random_forest_regressor_base_model()
        
        grid_search_cross_validator = GridSearchCV(
            base_model,
            hyperparameter_grid,
            cv=cross_validation_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search_cross_validator.fit(input_features_train, output_targets_train)
        
        print(f"Best Random Forest hyperparameters found: {grid_search_cross_validator.best_params_}")
        
        best_trained_model = grid_search_cross_validator.best_estimator_
        
        return best_trained_model


class ModelPredictor:
    
    @staticmethod
    def predict_with_normalization(trained_model, input_features, feature_scaler, target_scaler):
        
        input_features_normalized = feature_scaler.transform(input_features)
        predicted_output_normalized = trained_model.predict(input_features_normalized)
        predicted_output = target_scaler.inverse_transform(predicted_output_normalized)
        
        return predicted_output
    
    @staticmethod
    def predict_without_normalization(trained_model, input_features):
        
        predicted_output = trained_model.predict(input_features)
        
        return predicted_output
    
    @staticmethod
    def predict_single_sample_with_normalization(trained_model, input_features_single, 
                                                 feature_scaler, target_scaler):
        
        if input_features_single.ndim == 1:
            input_features_single = input_features_single.reshape(1, -1)
        
        predicted_output = ModelPredictor.predict_with_normalization(
            trained_model,
            input_features_single,
            feature_scaler,
            target_scaler
        )
        
        return predicted_output.flatten()
    
    @staticmethod
    def predict_single_sample_without_normalization(trained_model, input_features_single):
        
        if input_features_single.ndim == 1:
            input_features_single = input_features_single.reshape(1, -1)
        
        predicted_output = ModelPredictor.predict_without_normalization(
            trained_model,
            input_features_single
        )
        
        return predicted_output.flatten()
