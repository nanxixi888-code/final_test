import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


class DataLoader:
    
    @staticmethod
    def load_pickle_file(pickle_filename):
        with open(pickle_filename, 'rb') as file_handle:
            loaded_data = pickle.load(file_handle)
        return loaded_data
    
    @staticmethod
    def save_pickle_file(data_to_save, output_filename):
        with open(output_filename, 'wb') as file_handle:
            pickle.dump(data_to_save, file_handle)
        print(f"Data saved to {output_filename}")


class DatasetSplitter:
    
    @staticmethod
    def split_train_validation_test(input_features, output_targets, 
                                     training_size=0.7, 
                                     validation_size=0.15, 
                                     test_size=0.15, 
                                     random_seed=42):
        
        assert abs(training_size + validation_size + test_size - 1.0) < 1e-6
        
        input_features_train_validation, input_features_test, output_targets_train_validation, output_targets_test = train_test_split(
            input_features, output_targets, test_size=test_size, random_state=random_seed
        )
        
        validation_size_adjusted = validation_size / (training_size + validation_size)
        
        input_features_train, input_features_validation, output_targets_train, output_targets_validation = train_test_split(
            input_features_train_validation, output_targets_train_validation, 
            test_size=validation_size_adjusted, random_state=random_seed
        )
        
        return (input_features_train, input_features_validation, input_features_test,
                output_targets_train, output_targets_validation, output_targets_test)


class FeatureNormalizer:
    
    @staticmethod
    def normalize_with_standard_scaler(data_train, data_validation, data_test):
        scaler = StandardScaler()
        
        data_train_normalized = scaler.fit_transform(data_train)
        data_validation_normalized = scaler.transform(data_validation)
        data_test_normalized = scaler.transform(data_test)
        
        return data_train_normalized, data_validation_normalized, data_test_normalized, scaler
    
    @staticmethod
    def fit_scaler_on_training_data(data_train):
        scaler = StandardScaler()
        data_train_normalized = scaler.fit_transform(data_train)
        return data_train_normalized, scaler
    
    @staticmethod
    def apply_existing_scaler(data, scaler):
        return scaler.transform(data)


class ModelEvaluator:
    
    @staticmethod
    def compute_regression_metrics(actual_values, predicted_values):
        mean_squared_error_value = mean_squared_error(actual_values, predicted_values)
        root_mean_squared_error_value = np.sqrt(mean_squared_error_value)
        mean_absolute_error_value = mean_absolute_error(actual_values, predicted_values)
        r_squared_score = r2_score(actual_values, predicted_values)
        
        return {
            'mse': mean_squared_error_value,
            'rmse': root_mean_squared_error_value,
            'mae': mean_absolute_error_value,
            'r2': r_squared_score
        }
    
    @staticmethod
    def evaluate_model_with_metrics(trained_model, input_features, output_targets, 
                                    dataset_name, target_scaler=None, is_normalized=False):
        
        if is_normalized and target_scaler is not None:
            predicted_targets_normalized = trained_model.predict(input_features)
            predicted_targets = target_scaler.inverse_transform(predicted_targets_normalized)
            actual_targets = target_scaler.inverse_transform(output_targets)
        else:
            predicted_targets = trained_model.predict(input_features)
            actual_targets = output_targets
        
        metrics = ModelEvaluator.compute_regression_metrics(actual_targets, predicted_targets)
        
        print(f"\n{dataset_name} Dataset Evaluation:")
        print(f"  Mean Squared Error (MSE): {metrics['mse']:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
        print(f"  Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        print(f"  R-squared Score (R2): {metrics['r2']:.6f}")
        
        return predicted_targets, actual_targets, metrics
    
    @staticmethod
    def print_metrics_comparison_table(metrics_dict_list, model_names, dataset_names):
        
        print("\n" + "=" * 100)
        print("Model Performance Comparison")
        print("=" * 100)
        
        for model_index, model_name in enumerate(model_names):
            print(f"\n{model_name} Performance:")
            print("-" * 100)
            print(f"{'Dataset':<15} {'MSE':<15} {'RMSE':<15} {'MAE':<15} {'R2':<15}")
            print("-" * 100)
            
            for dataset_index, dataset_name in enumerate(dataset_names):
                metrics = metrics_dict_list[model_index][dataset_index]
                print(f"{dataset_name:<15} {metrics['mse']:<15.6f} {metrics['rmse']:<15.6f} {metrics['mae']:<15.6f} {metrics['r2']:<15.6f}")
        
        print("\n" + "=" * 100)


class ModelPersistence:
    
    @staticmethod
    def save_model_with_scalers(trained_model, feature_scaler, target_scaler, model_filename):
        
        model_package = {
            'model': trained_model,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }
        
        joblib.dump(model_package, model_filename)
        print(f"\nTrained model and scalers saved to {model_filename}")
    
    @staticmethod
    def save_model_only(trained_model, model_filename):
        
        joblib.dump(trained_model, model_filename)
        print(f"\nTrained model saved to {model_filename}")
    
    @staticmethod
    def load_model_with_scalers(model_filename):
        
        model_package = joblib.load(model_filename)
        return model_package['model'], model_package['feature_scaler'], model_package['target_scaler']
    
    @staticmethod
    def load_model_only(model_filename):
        
        trained_model = joblib.load(model_filename)
        return trained_model


class VisualizationPlotter:
    
    @staticmethod
    def plot_scatter_with_diagonal(actual_values, predicted_values, xlabel, ylabel, title):
        
        plt.scatter(actual_values, predicted_values, alpha=0.5)
        plt.plot([actual_values.min(), actual_values.max()],
                 [actual_values.min(), actual_values.max()],
                 'r--', lw=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
    
    @staticmethod
    def plot_residual_histogram(actual_values, predicted_values, title):
        
        residuals = actual_values - predicted_values
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residual Error')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True)
    
    @staticmethod
    def create_figure_with_subplots(figure_size, num_rows, num_cols):
        
        return plt.figure(figsize=figure_size)
    
    @staticmethod
    def save_and_close_figure(filename):
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


class ArrayOperations:
    
    @staticmethod
    def clip_values_to_limits(values, lower_limits, upper_limits):
        
        return np.clip(values, lower_limits, upper_limits)
    
    @staticmethod
    def compute_euclidean_distance(point_a, point_b):
        
        return np.linalg.norm(point_a - point_b)
    
    @staticmethod
    def concatenate_arrays_horizontally(array_list):
        
        return np.concatenate(array_list, axis=1)
    
    @staticmethod
    def split_array_into_parts(array, split_indices):
        
        return np.split(array, split_indices, axis=1)

