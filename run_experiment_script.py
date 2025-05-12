import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import functions from our modules
from combined_main_script import (
    create_sepia_dataset,
    load_dataset,
    train_ann_model,
    train_cnn_model,
    compare_models
)

from hyperparameter_tuning_script import (
    tune_ann,
    tune_cnn,
    ANN_PARAM_GRID,
    CNN_PARAM_GRID
)

def setup_experiment_directory(args):
    """
    Set up experiment directory
    
    Args:
        args: Command line arguments
    
    Returns:
        experiment_dir: Path to experiment directory
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"sepia_classification_{timestamp}"
    
    if args.experiment_name:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join("experiments", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    
    # Change working directory to experiment directory
    os.chdir(experiment_dir)
    
    return experiment_dir

def save_experiment_config(args):
    """
    Save experiment configuration
    
    Args:
        args: Command line arguments
    """
    with open("experiment_config.txt", "w") as f:
        f.write("Experiment Configuration:\n")
        f.write("=" * 50 + "\n")
        
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        f.write("\n")
        f.write("System Information:\n")
        f.write("=" * 50 + "\n")
        f.write(f"TensorFlow Version: {tf.__version__}\n")
        f.write(f"GPU Available: {tf.config.list_physical_devices('GPU') != []}\n")
        f.write(f"GPU Devices: {tf.config.list_physical_devices('GPU')}\n")

def run_experiment(args):
    """
    Run the complete sepia classification experiment
    
    Args:
        args: Command line arguments
    """
    # Set up experiment directory
    experiment_dir = setup_experiment_directory(args)
    print(f"Experiment directory: {experiment_dir}")
    
    # Save experiment configuration
    save_experiment_config(args)
    
    # Print welcome message
    print("=" * 80)
    print(f"SEPIA CLASSIFICATION EXPERIMENT: {os.path.basename(experiment_dir)}")
    print("=" * 80)
    
    # Step 1: Create dataset
    if args.source_folder:
        print("\n===== Step 1: Creating Dataset =====")
        create_sepia_dataset(
            args.source_folder,
            args.num_samples,
            (args.img_size, args.img_size)
        )
    
    # Step 2: Load datasets
    print("\n===== Step 2: Loading Datasets =====")
    X_ann, y_ann = load_dataset(flatten_for_ann=True, img_size=(args.img_size, args.img_size))
    X_cnn, y_cnn = load_dataset(flatten_for_ann=False, img_size=(args.img_size, args.img_size))
    
    # Step 3: Hyperparameter tuning (if enabled)
    ann_best_params = None
    cnn_best_params = None
    
    if args.do_hyperparameter_tuning:
        print("\n===== Step 3: Hyperparameter Tuning =====")
        
        print("\nTuning ANN hyperparameters...")
        ann_best, _ = tune_ann(
            X_ann, y_ann,
            ANN_PARAM_GRID,
            epochs=args.max_tuning_epochs,
            max_combinations=args.max_combinations
        )
        ann_best_params = ann_best['hyperparams']
        ann_batch_size = ann_best['batch_size']
        
        print("\nTuning CNN hyperparameters...")
        cnn_best, _ = tune_cnn(
            X_cnn, y_cnn,
            CNN_PARAM_GRID,
            epochs=args.max_tuning_epochs,
            max_combinations=args.max_combinations
        )
        cnn_best_params = cnn_best['hyperparams']
        cnn_batch_size = cnn_best['batch_size']
    else:
        # Use default hyperparameters
        ann_batch_size = 32
        cnn_batch_size = 32
    
    # Step 4: Train models
    print("\n===== Step 4: Training Models =====")
    
    print("\nTraining ANN model...")
    ann_model, ann_history, ann_test_data = train_ann_model(
        X_ann, y_ann,
        hyperparams=ann_best_params,
        epochs=args.ann_epochs,
        batch_size=ann_batch_size
    )
    
    print("\nTraining CNN model...")
    cnn_model, cnn_history, cnn_test_data = train_cnn_model(
        X_cnn, y_cnn,
        hyperparams=cnn_best_params,
        epochs=args.cnn_epochs,
        batch_size=cnn_batch_size,
        use_augmentation=args.use_augmentation
    )
    
    # Save models
    ann_model.save(os.path.join("models", "best_ann_model.h5"))
    cnn_model.save(os.path.join("models", "best_cnn_model.h5"))
    
    # Step 5: Compare models
    print("\n===== Step 5: Comparing Models =====")
    best_model = compare_models(
        (ann_model, ann_history, ann_test_data),
        (cnn_model, cnn_history, cnn_test_data)
    )
    
    # Step 6: Generate final summary
    print("\n===== Step 6: Generating Summary =====")
    
    # Save summary to file
    with open(os.path.join("results", "experiment_summary.txt"), "w") as f:
        f.write("Sepia Classification Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"- Total images: {len(X_ann)}\n")
        f.write(f"- Image size: {args.img_size}x{args.img_size}\n\n")
        
        f.write("ANN Model:\n")
        f.write(f"- Parameters: {ann_model.count_params():,}\n")
        f.write(f"- Training epochs: {len(ann_history.history['loss'])}\n")
        f.write(f"- Final validation accuracy: {ann_history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"- Final validation loss: {ann_history.history['val_loss'][-1]:.4f}\n\n")
        
        f.write("CNN Model:\n")
        f.write(f"- Parameters: {cnn_model.count_params():,}\n")
        f.write(f"- Training epochs: {len(cnn_history.history['loss'])}\n")
        f.write(f"- Final validation accuracy: {cnn_history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"- Final validation loss: {cnn_history.history['val_loss'][-1]:.4f}\n\n")
        
        f.write("Conclusion:\n")
        if ann_history.history['val_accuracy'][-1] > cnn_history.history['val_accuracy'][-1]:
            f.write("The ANN model performed better for this task.\n")
        else:
            f.write("The CNN model performed better for this task.\n")
    
    print("\n===== Experiment Completed =====")
    print(f"All results saved to: {experiment_dir}")
    
    return experiment_dir

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Complete Sepia Classification Experiment")
    
    # Dataset arguments
    parser.add_argument("--source_folder", type=str, default=None,
                        help="Path to folder with source images")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Maximum number of image pairs to create")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Size to resize images to")
    
    # Hyperparameter tuning arguments
    parser.add_argument("--do_hyperparameter_tuning", action="store_true",
                        help="Whether to perform hyperparameter tuning")
    parser.add_argument("--max_tuning_epochs", type=int, default=20,
                        help="Maximum number of epochs for hyperparameter tuning")
    parser.add_argument("--max_combinations", type=int, default=10,
                        help="Maximum number of hyperparameter combinations to try")
    
    # Training arguments
    parser.add_argument("--ann_epochs", type=int, default=50,
                        help="Maximum number of epochs for ANN training")
    parser.add_argument("--cnn_epochs", type=int, default=100,
                        help="Maximum number of epochs for CNN training")
    parser.add_argument("--use_augmentation", action="store_true",
                        help="Whether to use data augmentation for CNN")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment (default: sepia_classification_TIMESTAMP)")
    
    args = parser.parse_args()
    
    # Run experiment
    experiment_dir = run_experiment(args)
