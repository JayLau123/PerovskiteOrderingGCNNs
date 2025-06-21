#!/usr/bin/env python3
"""
Training script for Perovskite Ordering GCNNs using Weights & Biases (wandb)
This script replaces the SigOpt functionality with wandb for hyperparameter optimization.

Usage:
    python train_with_wandb.py --model_type CGCNN --struct_type unrelaxed --gpu_num 0 --obs_budget 1
"""

import argparse
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.run_wandb_experiment import run_wandb_experiment, run_single_training_run


def main():
    parser = argparse.ArgumentParser(description='Train GCNNs using wandb for hyperparameter optimization')
    parser.add_argument('--struct_type', default='unrelaxed', type=str, 
                       choices=['unrelaxed', 'relaxed', 'M3Gnet_relaxed'],
                       help='Structure representation type')
    parser.add_argument('--model_type', default='CGCNN', type=str,
                       choices=['CGCNN', 'e3nn', 'Painn'],
                       help='Model architecture type')
    parser.add_argument('--gpu_num', default=0, type=int,
                       help='GPU number to use')
    parser.add_argument('--obs_budget', default=1, type=int,
                       help='Number of hyperparameter optimization trials')
    parser.add_argument('--training_fraction', default=0.125, type=float,
                       help='Fraction of training data to use')
    parser.add_argument('--training_seed', default=0, type=int,
                       help='Random seed for training data selection')
    parser.add_argument('--data_name', default='data/', type=str,
                       help='Data directory path')
    parser.add_argument('--target_prop', default='dft_e_hull', type=str,
                       help='Target property to predict')
    parser.add_argument('--interpolation', default=False, action='store_true',
                       help='Use interpolation')
    parser.add_argument('--contrastive_weight', default=1.0, type=float,
                       help='Weight for contrastive loss')
    parser.add_argument('--project_name', default='perovskite-ordering-gcnns', type=str,
                       help='wandb project name')
    parser.add_argument('--single_run', action='store_true',
                       help='Run single training instead of hyperparameter sweep')
    
    args = parser.parse_args()
    
    print(f"Starting training with wandb...")
    print(f"Model: {args.model_type}")
    print(f"Structure type: {args.struct_type}")
    print(f"GPU: {args.gpu_num}")
    print(f"Training fraction: {args.training_fraction}")
    print(f"Target property: {args.target_prop}")
    print(f"Project: {args.project_name}")
    
    if args.single_run:
        print("Running single training run with default hyperparameters...")
        val_loss = run_single_training_run(
            struct_type=args.struct_type,
            model_type=args.model_type,
            gpu_num=args.gpu_num,
            training_fraction=args.training_fraction,
            data_name=args.data_name,
            target_prop=args.target_prop,
            interpolation=args.interpolation,
            contrastive_weight=args.contrastive_weight,
            training_seed=args.training_seed,
            project_name=args.project_name
        )
        print(f"Training completed. Final validation loss: {val_loss}")
    else:
        print(f"Running hyperparameter optimization with {args.obs_budget} trials...")
        run_wandb_experiment(
            struct_type=args.struct_type,
            model_type=args.model_type,
            gpu_num=args.gpu_num,
            obs_budget=args.obs_budget,
            training_fraction=args.training_fraction,
            data_name=args.data_name,
            target_prop=args.target_prop,
            interpolation=args.interpolation,
            contrastive_weight=args.contrastive_weight,
            training_seed=args.training_seed,
            project_name=args.project_name
        )
        print("Hyperparameter optimization completed!")
    
    print("Check your wandb dashboard for detailed results and visualizations.")


if __name__ == "__main__":
    main() 