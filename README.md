# Glass Type Prediction

This repository contains a machine learning project for predicting the type of glass based on its chemical composition. The project uses various machine learning algorithms to classify glass into different categories.
After identifing the optimal model this project finally uses the RNN model with the training accuracy of 0.91 .

## Project Structure


- **templates/**: Contains HTML templates for the web application.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **TestingWithAlgorithms.ipynb**: Jupyter notebook for testing various machine learning algorithms.
- **app.py**: Flask web application for predicting glass types.
- **glass.csv**: Dataset used for training and testing the model.
- **requirements.txt**: Lists the dependencies required for the project.
- **rnn_model.keras**: Saved Keras model for glass type prediction.

## Algorithms Used
The project explores various machine learning algorithms such as KNN, Naive Bayes, CNN, RNN ect,.. to find the best model for glass type prediction. Details and performance of these algorithms are documented in the TestingWithAlgorithms.ipynb notebook.

## Dataset
The dataset (glass.csv) contains chemical composition data of glass samples. The target variable is the type of glass, which is classified into the following categories:
Building Windows (Float Processed)
Building Windows (Non-Float Processed)
Vehicle Windows (Float Processed)
Vehicle Windows (Non-Float Processed)
Containers
Tableware
Headlamps

## Screenshots

### Accuracy
![Accuracy](images/Accuracy.png)

### Form page
![Form page](images/Form.png)

### Predicticed output page
![Predicticed output page](images/output.png)



