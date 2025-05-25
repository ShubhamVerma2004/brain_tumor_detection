import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_images(folder):
    classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(folder, class_name)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Class folder {class_name} not found")
            
        print(f"Loading {class_name} images...")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = img / 255.0
                images.append(img)
                labels.append(class_idx)
    
    return np.array(images), np.array(labels)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

def main():
    dataset_path = 'dataset'
    try:
        X, y = load_images(dataset_path)
        print(f"Loaded {len(X)} images")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = build_model((128, 128, 3), 4)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(
            X_train, y_train,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stop]
        )
        
        os.makedirs('models', exist_ok=True)
        save_model(model, 'models/brain_tumor_cnn.h5')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == '__main__':
    main()