# part1_torque_prediction.py

import numpy as np
from pathlib import Path
from typing import List

# --- 教授提供的工具 ---
# 确保 rollout_loader.py 和这个脚本在同一个文件夹下
from rollout_loader import load_rollouts, Rollout 

# --- 你自己的工具 ---
# 我们不再需要 DataLoader，但我们保留其他的
from common_utilities import DatasetSplitter, FeatureNormalizer, ModelEvaluator, ModelPersistence
from machine_learning_utilities import DatasetPreparator, MLPRegressorTrainer
from visualization_utilities import TorquePredictionVisualizer


class TorquePredictionPipeline:
    
    # 1. 修改 __init__: 不再需要 data_filename
    def __init__(self):
        
        self.input_features = None
        self.output_targets = None
        self.splits = {}
        self.normalized_data = {}
        self.scalers = {}
        self.trained_model = None
        self.evaluation_results = {}
        
    # 2. 修改 load_and_prepare_dataset: 使用 rollout_loader
    def load_and_prepare_dataset(self):
        
        print(f"\nLoading data from rollouts...")
        
        SCRIPT_DIR = Path(__file__).resolve().parent
        rollout_indices = [0, 1, 2, 3] # 对应 data_0.pkl...data_3.pkl
        
        try:
            # 调用教授的函数来加载数据
            loaded_rollouts = load_rollouts(indices=rollout_indices, directory=SCRIPT_DIR)
        except FileNotFoundError:
            print(f"Error: 找不到数据文件 (例如: data_0.pkl) 在 {SCRIPT_DIR}")
            print("请先运行 data_generator.py 来生成数据。")
            return self # 停止执行

        print(f"Successfully loaded {len(loaded_rollouts)} rollout files.")

        print("Preparing dataset for torque prediction...")
        
        # 3. 将加载好的数据列表 (List[Rollout]) 传递给 DatasetPreparator
        self.input_features, self.output_targets = DatasetPreparator.extract_torque_prediction_features_and_targets(
            loaded_rollouts
        )
        
        if self.input_features is None or self.input_features.size == 0:
            print("错误: 数据集准备失败。请检查 DatasetPreparator 的逻辑。")
            return self
            
        print(f"Input features shape: {self.input_features.shape}")
        print(f"Output targets shape: {self.output_targets.shape}")
        
        return self
    
    # --- 你所有其他的方法 (split_dataset..., normalize..., train...) 都保持原样 ---
    # --- 它们会从 common_utilities.py 和 visualization_utilities.py 中正确调用 ---
    
    def split_dataset_into_train_validation_test(self, training_size=0.7, validation_size=0.15, test_size=0.15):
        print("\nSplitting dataset into training, validation, and test sets...")
        (input_features_train, input_features_validation, input_features_test,
         output_targets_train, output_targets_validation, output_targets_test) = DatasetSplitter.split_train_validation_test(
            self.input_features, self.output_targets, training_size, validation_size, test_size)
        self.splits = {
            'input_train': input_features_train, 'input_validation': input_features_validation, 'input_test': input_features_test,
            'output_train': output_targets_train, 'output_validation': output_targets_validation, 'output_test': output_targets_test
        }
        print(f"Training set size: {input_features_train.shape[0]}")
        print(f"Validation set size: {input_features_validation.shape[0]}")
        print(f"Test set size: {input_features_test.shape[0]}")
        return self
    
    def normalize_features_and_targets(self):
        print("\nNormalizing input features...")
        (input_train_norm, input_val_norm, input_test_norm, feature_scaler) = FeatureNormalizer.normalize_with_standard_scaler(
            self.splits['input_train'], self.splits['input_validation'], self.splits['input_test'])
        print("Normalizing output targets...")
        (output_train_norm, output_val_norm, output_test_norm, target_scaler) = FeatureNormalizer.normalize_with_standard_scaler(
            self.splits['output_train'], self.splits['output_validation'], self.splits['output_test'])
        self.normalized_data = {
            'input_train': input_train_norm, 'input_validation': input_val_norm, 'input_test': input_test_norm,
            'output_train': output_train_norm, 'output_validation': output_val_norm, 'output_test': output_test_norm
        }
        self.scalers = {'feature_scaler': feature_scaler, 'target_scaler': target_scaler}
        return self
    
    def train_mlp_model_with_hyperparameter_search(self):
        print("\nTraining MLP Regressor with hyperparameter tuning...")
        print("This may take several minutes...")
        self.trained_model = MLPRegressorTrainer.train_with_grid_search_cross_validation(
            self.normalized_data['input_train'], self.normalized_data['output_train'])
        return self
    
    def evaluate_model_on_all_datasets(self):
        print("\nEvaluating model on different datasets...")
        predicted_train, actual_train, metrics_train = ModelEvaluator.evaluate_model_with_metrics(
            self.trained_model, self.normalized_data['input_train'], self.normalized_data['output_train'],
            "Training", self.scalers['target_scaler'], is_normalized=True)
        predicted_validation, actual_validation, metrics_validation = ModelEvaluator.evaluate_model_with_metrics(
            self.trained_model, self.normalized_data['input_validation'], self.normalized_data['output_validation'],
            "Validation", self.scalers['target_scaler'], is_normalized=True)
        predicted_test, actual_test, metrics_test = ModelEvaluator.evaluate_model_with_metrics(
            self.trained_model, self.normalized_data['input_test'], self.normalized_data['output_test'],
            "Test", self.scalers['target_scaler'], is_normalized=True)
        self.evaluation_results = {
            'train': {'predicted': predicted_train, 'actual': actual_train, 'metrics': metrics_train},
            'validation': {'predicted': predicted_validation, 'actual': actual_validation, 'metrics': metrics_validation},
            'test': {'predicted': predicted_test, 'actual': actual_test, 'metrics': metrics_test}
        }
        return self
    
    def generate_visualization_plots(self, number_of_joints=7):
        print("\nGenerating visualizations...")
        TorquePredictionVisualizer.plot_all_joints_predictions(
            self.evaluation_results['test']['predicted'], self.evaluation_results['test']['actual'],
            "Test", number_of_joints, save_prefix='part1_torque_prediction')
        return self
    
    def save_trained_model_to_file(self, output_filename='part1_torque_prediction_model.pkl'):
        ModelPersistence.save_model_with_scalers(
            self.trained_model, self.scalers['feature_scaler'],
            self.scalers['target_scaler'], output_filename)
        return self


def main():
    
    print("Part 1: Torque Command Prediction using MLP")
    print("=" * 60)
    
    # 4. 修改 main 函数: 调用 __init__ 时不再传入 data_filename
    torque_prediction_pipeline = TorquePredictionPipeline()
    
    torque_prediction_pipeline.load_and_prepare_dataset()
    
    # 5. 添加一个检查
    if torque_prediction_pipeline.input_features is None or torque_prediction_pipeline.input_features.size == 0:
        print("\nStopping pipeline due to data loading/preparation error.")
        return

    torque_prediction_pipeline.split_dataset_into_train_validation_test(
        training_size=0.7,
        validation_size=0.15,
        test_size=0.15
    )
    
    torque_prediction_pipeline.normalize_features_and_targets()
    
    torque_prediction_pipeline.train_mlp_model_with_hyperparameter_search()
    
    torque_prediction_pipeline.evaluate_model_on_all_datasets()
    
    torque_prediction_pipeline.generate_visualization_plots(number_of_joints=7)
    
    torque_prediction_pipeline.save_trained_model_to_file('part1_torque_prediction_model.pkl')
    
    print("\n" + "=" * 60)
    print("Part 1 completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
