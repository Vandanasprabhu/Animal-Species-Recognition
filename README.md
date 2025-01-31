# Animal Species Classification Using InceptionV3

## Overview
This project implements an **animal species classification model** using a pre-trained **InceptionV3** convolutional neural network (CNN). The model classifies **26 different animal species** from user-uploaded images with high accuracy.

## Features
- Uses **InceptionV3** as a feature extractor for transfer learning.
- Trained on a **custom dataset** stored in Google Drive.
- Image preprocessing with **ImageDataGenerator** for data augmentation.
- Model achieves **99.54% validation accuracy** in 5 epochs.
- Accepts user-uploaded images for real-time species classification.
- Displays **prediction results with confidence scores**.

## Technologies Used
- **Python**
- **TensorFlow/Keras**
- **NumPy**
- **PIL (Pillow)**
- **Matplotlib**
- **Google Colab**

## Dataset
The dataset consists of images classified into **26 animal species**. It is stored on **Google Drive** and loaded using **ImageDataGenerator**.

## Model Architecture
- **Base Model:** InceptionV3 (pre-trained on ImageNet, with top layers removed)
- **Custom Layers:**
  - GlobalAveragePooling2D
  - Fully connected dense layer with **ReLU** activation
  - Output layer with **softmax** activation (26 classes)
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/animal-classifier.git
   cd animal-classifier
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib pillow
   ```
3. Mount Google Drive in Google Colab to access the dataset.
4. Run the Jupyter Notebook or Python script to train the model.

## Model Training
Train the model using the following command:
```python
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)
```

## Prediction
Upload an image and classify the species using:
```python
prediction, confidence = predict_animal("path/to/image.jpg")
print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
```

## Results
- **Validation Accuracy:** 99.54%
- **Loss:** 0.0187
- **Performance:** Efficient classification with high confidence.

## Future Enhancements
- Expand dataset to include more species.
- Implement a web-based interface for easy image uploads.
- Deploy the model as an API for real-time classification.

## License
This project is open-source under the **MIT License**.

## Author
Developed by **Vandana S Prabhu**,**Anjala T M**,**Anugraha O B**,**Preethika Murukesan**.

