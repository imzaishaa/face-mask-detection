# face-mask-detection
This project uses a Convolutional Neural Network (CNN) to automatically detect whether a person is wearing a face mask in an image. It includes image preprocessing, model training, evaluation (accuracy, F1-score, confusion matrix), and real-time predictions using a dataset from Kaggle.
Here's a full **`README.md`** file you can use for your Face Mask Detection project on GitHub:

This project uses **Convolutional Neural Networks (CNNs)** in Python to detect whether a person is wearing a face mask or not. It is designed to support real-time prediction and performance evaluation using deep learning techniques. The model is trained on a labeled dataset from Kaggle and achieves high accuracy on binary classification tasks.


 📌 Features

- ✅ Binary image classification: With Mask / Without Mask  
- 📦 Dataset loading from Kaggle  
- 🖼️ Image preprocessing and resizing  
- 🧠 CNN architecture for training  
- 📊 Model evaluation: accuracy, precision, recall, F1-score  
- 🔥 Confusion matrix visualization  
- 🧪 Real-time image prediction with OpenCV  

 📁 Dataset

The project uses the **Face Mask Dataset** from Kaggle:  
[🔗 https://www.kaggle.com/datasets/omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

Classes:
- `with_mask/`
- `without_mask/`

To use the dataset:
1. Download it from Kaggle.
2. Place `kaggle.json` in the root directory.
3. Use the following code to download the dataset in Colab:
```python
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d omkargurav/face-mask-dataset

## 🛠️ Technologies Used

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* scikit-learn
* OpenCV
* Google Colab
 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/yourusername/face-mask-detection.git
```

2. Upload `kaggle.json` to access the dataset (if using Colab).

3. Run the notebook or script to:

   * Preprocess images
   * Train the CNN model
   * Evaluate performance
   * Test on custom input

4. Predict a custom image:

```python
input_image_path = 'path_to_your_image.jpg'


## 📊 Evaluation Metrics

The model is evaluated using:

* ✅ Accuracy
* 📈 Precision, Recall, F1-Score
* 🔷 Confusion Matrix

You can visualize the confusion matrix using:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(Y_test, y_pred_labels)
sns.heatmap(cm, annot=True)

## 📷 Sample Output

Image: with_mask_2.jpg
Prediction: With Mask ✅

 📚 Appendix

* CNN Model Summary
* Training/Validation Accuracy and Loss Plots
* Confusion Matrix Output
* Classification Report




