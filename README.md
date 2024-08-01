# Aerial Scene Classification Using ResNet50

This project demonstrates the application of Convolutional Neural Networks (CNNs), specifically the ResNet50 architecture, for classifying aerial images from the UC Merced Land Use Dataset into 21 distinct classes.

## Abstract

This project explores the use of a pre-trained ResNet50 model, fine-tuned for aerial image classification. The process involves data augmentation, transfer learning, and comprehensive evaluation using various metrics and visualizations, achieving significant accuracy improvements.

## Introduction

The goal of this project is to adapt a pre-trained ResNet50 model for a custom classification task using the UC Merced Land Use Dataset. This involves:
- Data preprocessing and augmentation
- Splitting data into training, validation, and test subsets
- Defining and fine-tuning the model
- Evaluating the model’s performance
- Visualizing the results

## Data and Methodology

### Data Description

- **Dataset**: UC Merced Land Use Dataset with 21 classes, each containing 100 images.
- **Image Size**: Resized to 256x256 pixels.
- **Augmentation**: Applied color jittering to enhance dataset diversity without random flips and rotations.

### Methodology

#### Model Choice and Justification

- **Model**: ResNet50, chosen for its depth and performance in image classification tasks.
- **Transfer Learning**: Utilized pre-trained weights from ImageNet to improve convergence speed and performance.

#### Training and Fine-tuning

- Unfreezing the last 20 layers of ResNet50 for fine-tuning.
- Adding batch normalization and dropout layers to prevent overfitting.
- Using Stochastic Gradient Descent (SGD) with momentum and weight decay for stable convergence.
- Training for 50 epochs with early stopping to avoid overfitting.

#### Model Architecture

- Initial convolutional layers reduce input size with max pooling.
- Residual blocks with varying kernel sizes (1x1, 3x3, 1x1).
- Fully connected layers with batch normalization, ReLU activation, and dropout.

## Evaluation Metrics

- **Accuracy**: Overall model accuracy.
- **Confusion Matrix**: Detailed class-wise performance.
- **ROC-AUC Curves**: Performance metrics for each class.
- **Visualizations**: Feature maps and filter visualizations to interpret model decisions.

## Results and Discussion

- **Performance**: Achieved test accuracy of 92.06%.
- **Filter and Feature Maps**: Showed effective learning of image representations.
- **Data Augmentation**: Improved model generalization without significant bias.
- **Computational Efficiency**: Transfer learning reduced computational costs.

## Challenges and Adjustments

- Addressed zero-variance filters in deeper layers with enhanced normalization.
- Fine-tuned only the last few layers for computational efficiency.
- Adjusted augmentation strategies to avoid overfitting.

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
- Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
- Fawcett, T. (2006). An introduction to ROC analysis.
- UC Merced Land Use Dataset.

## Usage

To run the project, open the `environment_project.ipynb` notebook in Jupyter Notebook or JupyterLab and execute the cells in order. Ensure you have the necessary libraries installed, including TensorFlow, NumPy, and Matplotlib.

## Contributing

Contributions are welcome. Feel free to fork this repository and submit pull requests for improvements or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
