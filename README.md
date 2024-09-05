# Sentiment-Analysis-of-Yelp-Reviews

This project demonstrates the application of a deep learning model to perform sentiment analysis on the Yelp Polarity Reviews dataset. The primary goal is to classify Yelp reviews into two categories—positive and negative—using a binary classification approach. This report details the dataset characteristics, model architecture, training process, and results achieved.

## Dataset Description

The Yelp Polarity Reviews dataset originates from the Yelp Dataset Challenge 2015 and has been curated by Xiang Zhang. It contains a total of 598,000 reviews split into 560,000 training samples and 38,000 test samples. Reviews rated with 1 or 2 stars are labeled as negative (class 1), and those with 3 or 4 stars are labeled as positive (class 2). The data is provided in CSV format with two columns: the class index and the review text. This dataset is specifically designed for binary sentiment classification and provides a balanced classification challenge.

## Model Architecture

The model uses a pre-trained text embedding (`nnlm-en-dim50`) from TensorFlow Hub, which converts text inputs into 50-dimensional embedding vectors. This layer is followed by:
- A fully connected dense layer with 128 neurons and ReLU activation, designed to learn higher-order features from the text embeddings.
- A final output layer with a single neuron and sigmoid activation to perform binary classification.

This architecture is chosen for its simplicity and effectiveness in handling text data through transfer learning, where the pre-trained embeddings significantly reduce the need for learning from scratch.

## Training Process

### Environment Setup
The model is trained in a Google Colab environment with the following configurations:
- TensorFlow 2.x is used to leverage the latest features of the framework.
- The notebook is integrated with Google Drive for data storage and retrieval.
- Matplotlib settings are configured for clear visual output of training results.

### Execution
Training involves the following steps:
- Loading the Yelp Polarity Reviews dataset from TensorFlow Datasets (TFDS).
- Splitting the data into 70% training and 30% validation sets.
- Employing callbacks such as ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau to monitor training and make adjustments dynamically.
- The model is compiled with the Adam optimizer and binary crossentropy loss function.

### Hyperparameters
- **Batch Size**: 27
- **Epochs**: Up to 10, with early stopping based on validation loss.
- **Optimizer**: Adam with a learning rate of \(10^{-3}\).

## Results and Evaluation

The training process results in a final model with an accuracy of 92.69% on the training data. The learning curves indicate effective learning without significant overfitting. The learning rate adjustment observed in the curves suggests a dynamic approach to training, although typically the learning rate would decrease over time.

Validation accuracy remains high throughout training, suggesting that the model generalizes well on unseen data. The performance on the test set has not been explicitly mentioned in this discussion and should be evaluated to confirm the model's effectiveness on completely new data.

## Conclusions

The model demonstrates strong performance in classifying Yelp reviews into positive and negative categories. The use of pre-trained embeddings allows for efficient learning, requiring fewer epochs to reach high accuracy. Future improvements could include experimenting with more complex architectures, such as recurrent neural networks or transformers, to potentially capture more nuanced aspects of sentiment in text.

