
This project demonstrates a classification task on the Digits dataset using four different machine learning algorithms: K-Nearest Neighbors (KNN), Random Forest (RF), Support Vector Machine (SVM), and Artificial Neural Network (ANN). This repository includes data pre-processing, model training, evaluation, and metric comparisons between the models.

## Dataset

The dataset used is the **Digits** dataset from Scikit-Learn. It consists of images of handwritten digits from 0 to 9, each represented as an 8x8 image, and a corresponding label indicating the actual digit.

- **Target Shape**: 1797 labels
- **Data Shape**: 1797 samples with 64 features each
- **Images Shape**: 1797 samples with 8x8 pixel images

## Project Outline

1. **Data Loading and Visualization**
   - Load the dataset and visualize a sample image for verification.
   
2. **Data Splitting**
   - Split data into training (80%) and testing (20%) sets.

3. **Data Preprocessing**
   - Normalize pixel values to the range \([0, 1]\) using `MinMaxScaler`.

4. **Model Training and Evaluation**
   - Train four different models: Random Forest, Support Vector Machine (SVC), Neural Network (MLP), and K-Nearest Neighbors.
   - Evaluate models using accuracy, precision, recall, and confusion matrix.

5. **Metrics Comparison**
   - Compare models by plotting accuracy, precision, and recall metrics for both training and test datasets.

## Models

- **K-Nearest Neighbors (KNN)**: \( k = 8 \)
- **Random Forest (RF)**: Max depth of 128 with 256 estimators
- **Support Vector Classifier (SVC)**: Linear kernel
- **Artificial Neural Network (ANN)**: One hidden layer with 256 neurons, adaptive learning rate

## Results

Each model is evaluated based on the following metrics:
- **Accuracy**
- **Precision**
- **Recall**

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/digit-classification.git
   cd digit-classification
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the code:

   ```bash
   python main.py
   ```

## Usage

Each step in the process, including data loading, splitting, preprocessing, model training, and evaluation, is executed sequentially in `main.py`. The `calculate_metrics()` function computes and prints accuracy, precision, recall, and confusion matrix for both training and test datasets.

## Visualizations

The code includes visualizations to compare model metrics:
- **Training Accuracy Comparison**
- **Test Accuracy Comparison**
- **Precision Comparison**
- **Recall Comparison**

These bar plots enable a quick view of model performance across different metrics.

## Example Output

After running `main.py`, the output includes:
- A sample digit image.
- Metrics for each model: accuracy, precision, recall, and confusion matrix.
- Comparative bar plots of model metrics.

## License

This project is licensed under the MIT License.