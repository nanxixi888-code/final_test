import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class DatasetPreparator:
    
    @staticmethod
    def extract_torque_prediction_features_and_targets(loaded_data):
        
        measured_joint_positions = loaded_data['q_mes']
        desired_joint_positions = loaded_data['q_des']
        torque_commands = loaded_data['tau_cmd']
        
        joint_angle_error = desired_joint_positions - measured_joint_positions
        
        input_features = joint_angle_error
        output_targets = torque_commands
        
        return input_features, output_targets
    
    @staticmethod
    def extract_trajectory_prediction_features_and_targets(loaded_data):
        
        measured_joint_positions = loaded_data['q_mes']
        desired_joint_positions = loaded_data['q_des']
        desired_joint_velocities = loaded_data['qd_des']
        final_target_cartesian_positions = loaded_data['desired_cartesian_pos']
        
        input_features = np.concatenate([measured_joint_positions, final_target_cartesian_positions], axis=1)
        
        output_targets = np.concatenate([desired_joint_positions, desired_joint_velocities], axis=1)
        
        return input_features, output_targets
    
    @staticmethod
    def split_concatenated_output_into_positions_and_velocities(concatenated_output, number_of_joints):
        
        desired_joint_positions = concatenated_output[:number_of_joints]
        desired_joint_velocities = concatenated_output[number_of_joints:]
        
        return desired_joint_positions, desired_joint_velocities


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

