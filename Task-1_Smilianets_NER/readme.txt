Named Entity Recognition (NER) Model for Mountain Names
Overview
This repository contains the implementation of a Named Entity Recognition (NER) model designed to identify mountain names within texts. The model is based on the DistilBERT architecture and has been trained on a custom dataset.

Project Structure
The project is organized into the following components:

ner_model: This directory contains the trained model weights. You can find the pre-trained DistilBERT model for token classification along with additional fine-tuned weights specific to the mountain names dataset.

ner_tokenizer: This directory includes the tokenizer used to preprocess the text data for training and inference. It is essential for tokenizing input text before passing it to the model.

Model_Training.py: This script provides the code for training the NER model. It reads the labeled dataset from Sentence_Data.csv, preprocesses the data, and fine-tunes the DistilBERT model for NER.

Model_Inference.py: This script is used for making predictions on new text data using the trained model. It loads the pre-trained model and tokenizer from the ner_model and ner_tokenizer directories, respectively, and performs inference on user-provided text.

Sentence_Data.csv: This CSV file contains the labeled dataset used for training the NER model. It includes columns for text and corresponding labels.

Task-1_Smilianets_Dataset_Creation.ipynb: This Jupyter notebook guides you through the process of dataset creation. It explains how the labeled dataset is generated and provides insights into the data preprocessing steps.

Task-1_Smilianets_demo.ipynb: This Jupyter notebook serves as a demonstration of the NER model in action. It includes code snippets for using the model to predict mountain names in sample text.

Getting Started
Follow these steps to set up and run the project:

Clone the repository:
git clone <repository-url>
cd <repository-directory>
Install the required libraries:

pip install -r requirements.txt
Run the Model_Training.py script to train the NER model:

python Model_Training.py
The script will save the trained model weights in the ner_model directory.

Run the Model_Inference.py script to make predictions:

python Model_Inference.py
This script loads the pre-trained model and tokenizer, allowing you to input text and receive predictions for mountain names.

Model Evaluation and Improvements
The training script (Model_Training.py) is configured for 20 epochs, with learning rate adjustment using a scheduler. However, model performance may benefit from further hyperparameter tuning or experimenting with different architectures.

For further improvements, consider exploring larger pre-trained models or adjusting the dataset to include a more diverse set of examples.

Conclusion
This project demonstrates the creation and utilization of an NER model for identifying mountain names in text. The provided scripts and notebooks offer transparency into the dataset creation process, training steps, and model inference. Feel free to explore, modify, and extend the code to suit your specific needs. Good luck!