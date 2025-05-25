from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__, template_folder='templates', static_folder='static')

# Tumor type mapping
TUMOR_TYPES = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "Pituitary Tumor",
    3: "No Tumor Detected"
}

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/brain_tumor_cnn.h5'
PLOTS_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def load_tumor_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        print(f"Loading model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        return None

model = load_tumor_model()

def generate_training_plots():
    try:
        history = np.load('my_history.npy', allow_pickle=True).item()
        
        epochs_to_plot = [10, 20, 30, 40, 50]
        all_accuracy_values = history['accuracy']
        all_loss_values = history['loss']
        accuracy_values = [history['accuracy'][epoch-1] for epoch in epochs_to_plot]
        loss_values = [history['loss'][epoch-1] for epoch in epochs_to_plot]
        final_accuracy = all_accuracy_values[-1]

        # Generate accuracy/loss plot
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
        plot1_path = os.path.join(PLOTS_FOLDER, 'training_plot.png')
        fig1.savefig(plot1_path)
        plt.close(fig1)

        # Generate table data
        df = pd.DataFrame({
            'Epoch': epochs_to_plot,
            'Accuracy': accuracy_values,
            'Loss': loss_values
        })
        table_path = os.path.join(PLOTS_FOLDER, 'training_table.png')
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.axis('off')
        table = ax2.table(cellText=df.values, 
                         rowLabels=df.index, 
                         colLabels=df.columns, 
                         cellLoc='center', 
                         loc='center')
        fig2.savefig(table_path)
        plt.close(fig2)

        return plot1_path, table_path, final_accuracy

    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return None, None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Save file
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            
            # Process image
            img = cv2.imread(filename)
            if img is None:
                return render_template('error.html', 
                                    message="Invalid image file. Please upload a valid MRI scan.")
            
            try:
                img = cv2.resize(img, (128, 128))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                
                if model:
                    pred = model.predict(img)
                    class_idx = np.argmax(pred[0])
                    tumor_type = TUMOR_TYPES[class_idx]
                    confidence = float(pred[0][class_idx])
                else:
                    tumor_type = "Model Not Loaded"
                    confidence = 0.0
                    
                return render_template('result.html',
                                    result=tumor_type,
                                    confidence=round(confidence*100, 2),
                                    image_path=filename)
            except Exception as e:
                return render_template('error.html',
                                    message=f"Processing error: {str(e)}")
    
    return render_template('index.html')

@app.route('/training')
def training():
    plot_path, table_path, final_accuracy = generate_training_plots()
    if plot_path and table_path:
        plot_url = url_for('static', filename='plots/training_plot.png').replace('\\', '/')
        table_url = url_for('static', filename='plots/training_table.png').replace('\\', '/')
        return render_template('training.html', 
                             plot_url=plot_url,
                             table_url=table_url,
                             final_accuracy=round(final_accuracy*100, 2))
    else:
        return render_template('error.html', 
                            message="Could not generate training plots. Please check if my_history.npy exists.")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)