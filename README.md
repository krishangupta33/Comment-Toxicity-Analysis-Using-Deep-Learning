
# Comment Toxicity Analysis

This repository contains the code to build a comment toxicity model using deep learning in Python. The objective is to analyze comments and classify them based on their toxicity.

## Dataset

The dataset used in this project is provided through the [Kaggle competition]([https://www.kaggle.com/](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)). It comprises a large number of Wikipedia comments which have been labeled by human raters for toxic behavior.

### Types of Toxicity:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Technologies Used

- **Tensorflow**: For building and training the deep learning model.
- **Numpy**: To handle numerical operations and data manipulation.
- **Deep Learning**: Used deep neural networks for building the toxicity model.
- **Streamlit**: For hosting and visualizing the model's predictions in a web application.

## Model Performance

The trained model achieved the following metrics:

- **Precision**: 0.7732
- **Recall**: 0.7162
- **Accuracy**: 0.4684

The model has been trained and saved in an `.h5` file format.

## Web Application

The trained model has been integrated into a web application built using Streamlit. You can access and test the model [here](https://comment-toxicity-analysis.streamlit.app/).

## Files in this Repository

- `Tensorflow Model.ipynb`: This Jupyter notebook contains the code for building, training, and evaluating the deep learning model.
- `Comment Toxicity Analysis.py`: This Python script is used for hosting the model on Streamlit.

## Contribution

Feel free to fork this repository, raise issues, or submit Pull Requests if you have suggestions or improvements.

