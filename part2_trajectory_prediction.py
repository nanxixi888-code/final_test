# part2_trajectory_prediction.py

import numpy as np
from pathlib import Path
from typing import List
from rollout_loader import load_rollouts, Rollout 

# --- 我们现在 *重新* 导入 DatasetSplitter ---
from common_utilities import DatasetSplitter, FeatureNormalizer, ModelEvaluator, ModelPersistence
from machine_learning_utilities import DatasetPreparator, MLPRegressorTrainer, RandomForestRegressorTrainer
from visualization_utilities import TrajectoryPredictionVisualizer


class TrajectoryPredictionPipeline:
    
    def __init__(self):
        
        self.input_features = None # <--- 恢复: 用来存 *合并后* 的总数据
        self.output_targets = None # <--- 恢复: 用来存 *合并后* 的总数据
        self.splits = {}
        self.mlp_normalized_data = {}
        self.mlp_scalers = {}
        self.mlp_model = None
        self.mlp_evaluation_results = {}
        self.rf_model = None
        self.rf_evaluation_results = {}
        
    # --- 已修改: 恢复到 Part 1 的“先合并”逻辑 ---
    def load_and_prepare_dataset(self):
        
        print(f"\nLoading data from rollouts...")
        
        SCRIPT_DIR = Path(__file__).resolve().parent
        rollout_indices = [0, 1, 2, 3] 
        
        try:
            loaded_rollouts = load_rollouts(indices=rollout_indices, directory=SCRIPT_DIR)
        except FileNotFoundError:
            print(f"Error: 找不到数据文件 (例如: data_0.pkl)。")
            print("请先(重新)运行我们修改后的 data_generator.py 来生成数据。")
            self.input_features = None # 设置失败标志
            return self
        except ValueError as e:
            print(f"Error: 加载数据失败。请确保你的 .pkl 文件是最新的。")
            print(f"详细错误: {e}")
            self.input_features = None # 设置失败标志
            return self

        print(f"Successfully loaded {len(loaded_rollouts)} rollout files.")
        
        print("Combining and preparing dataset for trajectory prediction...")

        # --- 已修改: 立即将所有 4 个 rollouts 合并成一个大数据集 ---
        self.input_features, self.output_targets = DatasetPreparator.extract_trajectory_prediction_features_and_targets(
            loaded_rollouts
        )
        
        if self.input_features is None or self.input_features.size == 0:
            print("错误：数据集准备失败。")
            print("请检查 machine_learning_utilities.py 中的 extract_trajectory_prediction_features_and_targets 函数。")
            self.input_features = None # 设置失败标志
            return self
            
        print(f"Total Input features shape (from all 4 rollouts): {self.input_features.shape}")
        print(f"Total Output targets shape (from all 4 rollouts): {self.output_targets.shape}")
        
        return self
    
    # --- 已修改: 恢复到 Part 1 的“随机拆分”逻辑 ---
    def split_dataset_into_train_validation_test(self, training_size=0.7, validation_size=0.15, test_size=0.15):
        
        print("\nSplitting *combined* dataset into training, validation, and test sets...")
        
        (input_features_train, input_features_validation, input_features_test,
         output_targets_train, output_targets_validation, output_targets_test) = DatasetSplitter.split_train_validation_test(
            self.input_features, 
            self.output_targets, 
            training_size, 
            validation_size, 
            test_size,
            random_seed=42 # <--- 使用你熟悉的随机种子
        )
        
        self.splits = {
            'input_train': input_features_train,
            'input_validation': input_features_validation,
            'input_test': input_features_test,
            'output_train': output_targets_train,
            'output_validation': output_targets_validation,
            'output_test': output_targets_test
        }
        
        print(f"Training set size: {input_features_train.shape[0]}")
        print(f"Validation set size: {input_features_validation.shape[0]}")
        print(f"Test set size: {input_features_test.shape[0]}")
        
        return self
    
    # --- 你组员的所有其他函数 (prepare_mlp..., train..., evaluate...) 都是完美的 ---
    # --- 我们不需要修改它们 ---
    
    def prepare_mlp_normalized_data(self):
        
        print("\n" + "=" * 60)
        print("Preparing Data for MLP Regressor")
        print("=" * 60)
        
        print("\nNormalizing input features for MLP...")
        (input_train_norm, input_val_norm, input_test_norm, 
         feature_scaler) = FeatureNormalizer.normalize_with_standard_scaler(
            self.splits['input_train'],
            self.splits['input_validation'],
            self.splits['input_test']
        )
        
        print("Normalizing output targets for MLP...")
        (output_train_norm, output_val_norm, output_test_norm, 
         target_scaler) = FeatureNormalizer.normalize_with_standard_scaler(
            self.splits['output_train'],
            self.splits['output_validation'],
            self.splits['output_test']
        )
        
        self.mlp_normalized_data = {
            'input_train': input_train_norm,
            'input_validation': input_val_norm,
            'input_test': input_test_norm,
            'output_train': output_train_norm,
            'output_validation': output_val_norm,
            'output_test': output_test_norm
        }
        
        self.mlp_scalers = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }
        
        return self
    
    def train_mlp_model(self):
        # ... (无变化) ...
        print("\nTraining MLP Regressor with hyperparameter tuning...")
        print("This may take several minutes...")
        self.mlp_model = MLPRegressorTrainer.train_with_grid_search_cross_validation(
            self.mlp_normalized_data['input_train'],
            self.mlp_normalized_data['output_train']
        )
        return self
    
    def evaluate_mlp_model(self):
        # ... (无变化) ...
        print("\nEvaluating MLP model on different datasets...")
        predicted_train, actual_train, metrics_train = ModelEvaluator.evaluate_model_with_metrics(
            self.mlp_model, self.mlp_normalized_data['input_train'], self.mlp_normalized_data['output_train'],
            "MLP Training", self.mlp_scalers['target_scaler'], is_normalized=True
        )
        predicted_validation, actual_validation, metrics_validation = ModelEvaluator.evaluate_model_with_metrics(
            self.mlp_model, self.mlp_normalized_data['input_validation'], self.mlp_normalized_data['output_validation'],
            "MLP Validation", self.mlp_scalers['target_scaler'], is_normalized=True
        )
        predicted_test, actual_test, metrics_test = ModelEvaluator.evaluate_model_with_metrics(
            self.mlp_model, self.mlp_normalized_data['input_test'], self.mlp_normalized_data['output_test'],
            "MLP Test", self.mlp_scalers['target_scaler'], is_normalized=True
        )
        self.mlp_evaluation_results = {
            'train': {'predicted': predicted_train, 'actual': actual_train, 'metrics': metrics_train},
            'validation': {'predicted': predicted_validation, 'actual': actual_validation, 'metrics': metrics_validation},
            'test': {'predicted': predicted_test, 'actual': actual_test, 'metrics': metrics_test}
        }
        return self
    
    def train_random_forest_model(self):
        # ... (无变化) ...
        print("\n" + "=" * 60)
        print("Training Random Forest Regressor")
        print("=" * 60)
        print("\nTraining Random Forest Regressor with hyperparameter tuning...")
        print("This may take several minutes...")
        self.rf_model = RandomForestRegressorTrainer.train_with_grid_search_cross_validation(
            self.splits['input_train'],
            self.splits['output_train']
        )
        return self
    
    def evaluate_random_forest_model(self):
        # ... (无变化) ...
        print("\nEvaluating Random Forest model on different datasets...")
        predicted_train, actual_train, metrics_train = ModelEvaluator.evaluate_model_with_metrics(
            self.rf_model, self.splits['input_train'], self.splits['output_train'],
            "Random Forest Training", is_normalized=False
        )
        predicted_validation, actual_validation, metrics_validation = ModelEvaluator.evaluate_model_with_metrics(
            self.rf_model, self.splits['input_validation'], self.splits['output_validation'],
            "Random Forest Validation", is_normalized=False
        )
        predicted_test, actual_test, metrics_test = ModelEvaluator.evaluate_model_with_metrics(
            self.rf_model, self.splits['input_test'], self.splits['output_test'],
            "Random Forest Test", is_normalized=False
        )
        self.rf_evaluation_results = {
            'train': {'predicted': predicted_train, 'actual': actual_train, 'metrics': metrics_train},
            'validation': {'predicted': predicted_validation, 'actual': actual_validation, 'metrics': metrics_validation},
            'test': {'predicted': predicted_test, 'actual': actual_test, 'metrics': metrics_test}
        }
        return self
    
    def print_model_comparison_summary(self):
        # ... (无变化) ...
        metrics_list = [
            [
                self.mlp_evaluation_results['train']['metrics'],
                self.mlp_evaluation_results['validation']['metrics'],
                self.mlp_evaluation_results['test']['metrics']
            ],
            [
                self.rf_evaluation_results['train']['metrics'],
                self.rf_evaluation_results['validation']['metrics'],
                self.rf_evaluation_results['test']['metrics']
            ]
        ]
        model_names = ['MLP Regressor', 'Random Forest Regressor']
        dataset_names = ['Training', 'Validation', 'Test']
        ModelEvaluator.print_metrics_comparison_table(metrics_list, model_names, dataset_names)
        return self
    
    def generate_comparison_visualizations(self, number_of_joints=7):
        # ... (无变化) ...
        print("\nGenerating comparison visualizations...")
        TrajectoryPredictionVisualizer.plot_comparison_for_all_joints(
            self.mlp_evaluation_results['test']['predicted'],
            self.rf_evaluation_results['test']['predicted'],
            self.mlp_evaluation_results['test']['actual'],
            number_of_joints,
            save_prefix='part2_trajectory_prediction'
        )
        return self
    
    def save_trained_models(self, mlp_filename='part2_trajectory_prediction_mlp_model.pkl',
                           rf_filename='part2_trajectory_prediction_rf_model.pkl'):
        # ... (无变化) ...
        ModelPersistence.save_model_with_scalers(
            self.mlp_model,
            self.mlp_scalers['feature_scaler'],
            self.mlp_scalers['target_scaler'],
            mlp_filename
        )
        ModelPersistence.save_model_only(
            self.rf_model,
            rf_filename
        )
        return self


def main():
    
    print("Part 2: Trajectory Prediction using MLP and Random Forest")
    print("=" * 60)
    
    # --- 已修改: 恢复到“先合并，再拆分”的逻辑 ---
    trajectory_prediction_pipeline = TrajectoryPredictionPipeline()
    
    trajectory_prediction_pipeline.load_and_prepare_dataset()
    
    if trajectory_prediction_pipeline.input_features is None:
        print("\nStopping pipeline due to data loading/preparation error.")
        return

    # --- 已修改: 恢复到“随机拆分”的逻辑 ---
    trajectory_prediction_pipeline.split_dataset_into_train_validation_test(
        training_size=0.7,
        validation_size=0.15,
        test_size=0.15
    )
    
    # --- 后续所有步骤都与你组员的原始代码完全相同 ---
    
    trajectory_prediction_pipeline.prepare_mlp_normalized_data()
    
    trajectory_prediction_pipeline.train_mlp_model()
    
    trajectory_prediction_pipeline.evaluate_mlp_model()
    
    trajectory_prediction_pipeline.train_random_forest_model()
    
    trajectory_prediction_pipeline.evaluate_random_forest_model()
    
    trajectory_prediction_pipeline.print_model_comparison_summary()
    
    trajectory_prediction_pipeline.generate_comparison_visualizations(number_of_joints=7)
    
    trajectory_prediction_pipeline.save_trained_models(
        mlp_filename='part2_trajectory_prediction_mlp_model.pkl',
        rf_filename='part2_trajectory_prediction_rf_model.pkl'
    )
    
    print("\n" + "=" * 60)
    print("Part 2 completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
