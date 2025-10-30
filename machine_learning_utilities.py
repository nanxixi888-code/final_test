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
        q_mes_all: List[List[float]]
        qd_mes_all: List[List[float]]
        tau_mes_all: List[List[float]]


class DatasetPreparator:
    
    # --- 这是修改后的函数 ---
    @staticmethod
    def extract_torque_prediction_features_and_targets(loaded_rollouts: List[Rollout]):
        
        # 1. 你的函数现在接收一个列表 (loaded_rollouts)，而不是一个字典
        
        all_input_features = []
        all_output_targets = []

        # 2. 遍历列表中的每一个 rollout (即 data_0.pkl, data_1.pkl...)
        for rollout in loaded_rollouts:
            
            # 3. 提取数据
            # 我们从 Rollout 对象中获取数据 (使用 . 属性, 而不是 ['key'])
            q_mes = np.array(rollout.q_mes_all)    # (N, 7) 当前角度  [cite: 18-20, 23, 198-203]
            qd_mes = np.array(rollout.qd_mes_all)   # (N, 7) 当前速度  [cite: 18, 21, 23, 198-203]
            tau_mes = np.array(rollout.tau_mes_all) # (N, 7) 当前力矩  [cite: 18, 22-23, 198-203]

            # 4. !! 定义新的特征 !!
            # 因为 q_des (目标角度) 在教授的 .pkl 文件中缺失了, 
            # 我们无法计算 "joint_angle_error" (你原来的代码)。
            # 我们使用 [角度, 速度] (共14维) 作为输入来替代。
            
            input_features = np.hstack((q_mes, qd_mes)) # (N, 14)
            
            # 输出保持不变：力矩
            output_targets = tau_mes                    # (N, 7)
            
            # 5. 将这个 rollout 的数据添加到总列表中
            all_input_features.append(input_features)
            all_output_targets.append(output_targets)
        
        # 6. 将4个 rollout 的数据垂直堆叠 (stack) 成一个巨大的数据集
        if not all_input_features:
            print("错误：没有从 rollouts 中提取到任何数据。")
            return None, None
            
        final_input_features = np.vstack(all_input_features)
        final_output_targets = np.vstack(all_output_targets)
        
        return final_input_features, final_output_targets
    
    # --- 这是你原来的 Part 2 函数, 它也需要用同样的方式修改 ---
    @staticmethod
    def extract_trajectory_prediction_features_and_targets(loaded_rollouts: List[Rollout]):
        
        print("警告: extract_trajectory_prediction_features_and_targets (for Part 2) 尚未适配。")
        print("它需要的数据 (q_des, qd_des) 同样在 .pkl 文件中缺失。")
        
        # 暂时返回空值，以避免你的旧代码出错
        return np.array([]), np.array([]) 
    
    @staticmethod
    def split_concatenated_output_into_positions_and_velocities(concatenated_output, number_of_joints):
        
        desired_joint_positions = concatenated_output[:number_of_joints]
        desired_joint_velocities = concatenated_output[number_of_joints:]
        
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
