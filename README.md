# CNN-Based-Malaria-Parasite-Detection-System
Designed an automated system for blood cell image analysis to achieve faster and more accurate malaria diagnosis. 
Implemented a CNN-based model to classify red blood cells as parasitic or normal, improving diagnostic precision, enabling early detection, and supporting healthcare professionals with efficient results.

## 1. Project description

Malaria remains a major global health challenge. This project combines image processing and machine learning to automatically identify parasitic and non-parasitic samples from microscopic blood-smear images. The system is built around CNN models and image-processing pipelines (preprocessing, segmentation, augmentation) to produce reliable, reproducible results for classification.


## 2. Objective

To design and implement a robust malarial parasite detection system that:

* Automates identification of infected vs. uninfected blood cells from microscope images.
* Uses image preprocessing and CNN-based classification.
* Compares and optimizes CNN architectures and training strategies to achieve the best possible accuracy and robustness.
* Produces a usable inference pipeline (single-image prediction) suitable for prototyping integration into diagnostic workflows.


## 3. Scope

* Use of open-source, labeled blood-smear datasets (e.g., the Kaggle NIH malaria dataset) for model development.
* Image preprocessing (noise removal, normalization), segmentation (Otsu + watershed), augmentation, feature extraction, CNN training, and evaluation.
* The model and pipeline are intended as a research/prototype tool and a basis for further clinical integration (requires validation and regulatory review before clinical use).


## 4. Key features

* Binary classification: Infected vs. Uninfected.
* Image preprocessing with morphological operations for noise reduction.
* Segmentation with Otsu thresholding and Watershed for separating overlapping cells.
* Data augmentation to improve generalization.
* CNN architecture tuned for 128×128 RGB inputs.
* Training, evaluation, and single-image prediction utilities.


## 5. Technologies

* Programming: Python
* Core libraries: TensorFlow / Keras, OpenCV, NumPy, scikit-learn
* Visualization: Matplotlib, Seaborn
* Platform: Jupyter Notebook / Google Colab


## 6. Dataset

* Source used in this project: Cell images for detecting malaria (Kaggle / NIH).
* NOTE: Do not add the full dataset to the repo. Provide download instructions in `data/README.md` and load data in Colab via Kaggle API or Google Drive.


## 7. Workflow (high level)

1. Dataset collection and organization (train / val / test).
2. Image preprocessing (grayscale conversion if needed, resizing, normalization, morphological opening).
3. Image segmentation (Otsu threshold → morphological cleanup → distance transform → watershed).
4. Data augmentation (rotation, shift, shear, zoom, horizontal flip, rescale).
5. CNN model building and compilation.
6. Model training and validation.
7. Evaluation (accuracy, precision, recall, F1, confusion matrix).
8. Single-image prediction / inference.


## 8. Module descriptions

### 8.1 Dataset preparation module

* Collect well-labeled blood-smear images (infected / uninfected).
* Organize folder structure: `data/train/infected`, `data/train/uninfected`, `data/val/...`, `data/test/...`.
* Add a `data/README.md` explaining how to download and place the dataset.

### 8.2 Data preprocessing module

* Resize images to a uniform size (128×128).
* Normalize pixel values (rescale = 1/255).
* Remove noise and small artifacts using morphological operations (opening).
  Example (OpenCV):

```python
import cv2
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
clean = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
```

### 8.3 Image segmentation module

* Use Otsu’s thresholding to create a binary map.
* Apply morphological closing/opening to remove small holes/noise.
* Use distance transform + watershed to separate overlapping cells.
  Typical watershed pipeline (conceptual):

```python
# after thresholding & noise removal
dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
markers = cv2.connectedComponents(np.uint8(sure_fg))[1]
markers = markers + 1
markers[unknown==255] = 0
markers = cv2.watershed(original_img, markers)
original_img[markers == -1] = [255,0,0]  # mark boundaries
```

### 8.4 Feature extraction module

* For this project, the CNN learns hierarchical features directly from image pixels (no manual hand-crafted features required).
* Input shape used: (128, 128, 3).

### 8.5 Classification (CNN) module

* Sequential CNN with three Conv2D + MaxPooling2D blocks and two fully connected layers (128, 1).
* Activation: ReLU for hidden layers, Sigmoid for output (binary classification).
* Loss: `binary_crossentropy`; Optimizer: Adam (lr = 0.001); Metric: accuracy.

Example Keras model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 8.6 Training and identification module

* Use `ImageDataGenerator` for augmentation and generator-based training for memory efficiency.
  Data augmentation parameters used:
* `rescale=1/255`
* `rotation_range=20`
* `width_shift_range=0.2`
* `height_shift_range=0.2`
* `shear_range=0.2`
* `zoom_range=0.2`
* `horizontal_flip=True`
* `fill_mode='nearest'`

Example:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory('data/train', target_size=(128,128), class_mode='binary', batch_size=32)
```

Training example:

```python
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=validation_generator)
```


## 9. Model architecture summary & math (brief)

* Convolution operation: sliding filters over image patches, producing feature maps (dot product + bias).
  z = (w · x) + b
* Activation: ReLU: f(x) = max(0, x)
* Pooling: Max pooling downsamples spatial dimensions.
* Dense layer: z = W·X + b, followed by activation.
* Output: Sigmoid for binary probability.

## 10. Evaluation & metrics

* Primary metrics: Accuracy, Precision, Recall, F1-score.
* Use a confusion matrix to inspect class-wise performance.
* Recommended: compute ROC-AUC when possible for robust performance insight.

Example confusion matrix:

```python
from sklearn.metrics import confusion_matrix, classification_report
y_pred = (model.predict(test_generator) > 0.5).astype("int32")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
```


## Conclusion

This project demonstrates how Convolutional Neural Networks (CNNs) can effectively classify blood smear images as parasitic or non-parasitic. By preparing the dataset, training the model, and validating its performance, the system achieves reliable accuracy for malaria detection. While the current approach is a strong baseline, future improvements like transfer learning or fine-tuning could further enhance performance.

