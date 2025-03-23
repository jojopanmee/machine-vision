import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
from yolov8 import*

def plot_training_metrics(results_file):
    """Plot training metrics from results CSV."""
    # Load training results
    results = pd.read_csv(results_file)
    
    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    axs[0, 0].plot(results['epoch'], results['train/box_loss'], label='train')
    axs[0, 0].plot(results['epoch'], results['val/box_loss'], label='val')
    axs[0, 0].set_title('Box Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Plot mAP50
    axs[0, 1].plot(results['epoch'], results['metrics/mAP50(B)'])
    axs[0, 1].set_title('mAP@0.5')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('mAP')
    
    # Plot mAP50-95
    axs[1, 0].plot(results['epoch'], results['metrics/mAP50-95(B)'])
    axs[1, 0].set_title('mAP@0.5:0.95')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('mAP')
    
    # Plot precision and recall
    axs[1, 1].plot(results['epoch'], results['metrics/precision(B)'], label='precision')
    axs[1, 1].plot(results['epoch'], results['metrics/recall(B)'], label='recall')
    axs[1, 1].set_title('Precision and Recall')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_metrics.png"))
    plt.close()

def class_performance_analysis(val_results, class_names):
    """Analyze and visualize per-class performance."""
    # Extract per-class metrics
    stats = val_results.box.results_dict
    
    # Create dataframe
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': stats['metrics/precision(B)'],
        'Recall': stats['metrics/recall(B)'],
        'mAP50': stats['metrics/mAP50(B)']
    })
    
    # Plot class metrics
    plt.figure(figsize=(10, 6))
    class_metrics.plot(x='Class', y=['Precision', 'Recall', 'mAP50'], kind='bar')
    plt.ylabel('Score')
    plt.title('Per-class Performance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, "class_performance.png"))
    plt.close()
    
    return class_metrics

    