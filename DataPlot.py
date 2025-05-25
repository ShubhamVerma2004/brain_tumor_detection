import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def generate_training_plots():
    # Load the history data
    history = np.load('C:Users/Acer/Desktop/brain_tumor_detection/my_history.npy', allow_pickle='TRUE').item()

    epochs_to_plot = [10, 20, 30, 40, 50]

    all_accuracy_values = history['accuracy']
    all_loss_values = history['loss']
    accuracy_values = [history['accuracy'][epoch-1] for epoch in epochs_to_plot]
    loss_values = [history['loss'][epoch-1] for epoch in epochs_to_plot]

    final_accuracy = all_accuracy_values[-1]

    # Create plot directory if it doesn't exist
    plot_dir = 'static/plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Generate and save the accuracy/loss plot
    fig1, axs1 = plt.subplots(2, 1, figsize=(8, 6))
    axs1[0].plot(range(len(all_accuracy_values)), all_accuracy_values, label='Accuracy')
    axs1[0].scatter(epochs_to_plot, accuracy_values, marker='o', color='b', label='Specific Epochs')  
    axs1[1].plot(range(len(all_loss_values)), all_loss_values, label='Loss')
    axs1[1].scatter(epochs_to_plot, loss_values, marker='o', color='r', label='Specific Epochs')  

    axs1[0].set_xlabel('Epoch')
    axs1[0].set_ylabel('Accuracy')
    axs1[0].set_title('Model Accuracy for VGG16')
    axs1[0].legend()
    axs1[0].grid(True)

    axs1[1].set_xlabel('Epoch')
    axs1[1].set_ylabel('Loss')
    axs1[1].set_title('Model Loss for VGG16')
    axs1[1].legend()
    axs1[1].grid(True)
    fig1.subplots_adjust(hspace=0.4)
    
    plot_path = os.path.join(plot_dir, 'training_plot.png')
    fig1.savefig(plot_path)
    plt.close(fig1)

    # Generate and save the table
    df = pd.DataFrame({
        'Epoch': epochs_to_plot,
        'Accuracy': accuracy_values,
        'Loss': loss_values
    })

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    table = ax2.table(cellText=df.values, 
                     rowLabels=df.index, 
                     colLabels=df.columns, 
                     cellLoc='center', 
                     loc='center') 
    ax2.axis('off')
    
    table_path = os.path.join(plot_dir, 'training_table.png')
    fig2.savefig(table_path)
    plt.close(fig2)

    return {
        'plot_path': plot_path,
        'table_path': table_path,
        'final_accuracy': final_accuracy
    }