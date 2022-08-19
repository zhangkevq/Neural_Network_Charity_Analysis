# Neural_Network_Charity_Analysis
Neural Networks and Deep Learning Models

## Environment
Tensorflow v. 2.4.1

## Overview 
1. Compare the differences between the traditional machine learning classification and regression models and the neural network models.
2. Describe the perceptron model and its components.
3. Implement neural network models using TensorFlow.
4. Explain how different neural network structures change algorithm performance.
5. Preprocess and construct datasets for neural network models.
6. Compare the differences between neural network models and deep neural networks.
7. Implement deep neural network models using TensorFlow.
8. Save trained TensorFlow models for later use.

## Purpose
A foundation, Alphabet Soup, wants to predict where to make investments.  The goal is to use machine learning and neural networks to apply features on a provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.  The initial file has 34,000 organizations and a number of columns that capture metadata about each organization from past successful fundings.

## Results:

### Data Preprocessing
1. What variable(s) are considered the target(s) for your model?    
Checking to see if the target is marked as successful in the DataFrame, indicating that it has been successfully funded by AlphabetSoup.  

2. What variable(s) are considered to be the features for your model?    
The IS_SUCCESSFUL column is the feature chosen for this dataset.

3. What variable(s) are neither targets nor features, and should be removed from the input data?    
The EIN and NAME columns will not increase the accuracy of the model and can be removed to improve code efficiency. 

### Compiling, Training, and Evaluating the Model
4. How many neurons, layers, and activation functions did you select for your neural network model, and why?    
In the optimized model, layer 1 started with 120 neurons with a relu activation.  For layer 2, it dropped to 80 neurons and continued with the relu activation.  From there, the sigmoid activation seemed to be the better fit for layers 3 (40 neurons) and layer 4 (20 neurons).    

![Pic 1](https://github.com/zhangkevq/Neural_Network_Charity_Analysis/blob/main/images/compile_train_evaluate.png)   

5. Were you able to achieve the target model performance?   
The target for the model was 75%, but the best the model could produce was 72.7%.

6. What steps did you take to try and increase model performance?   
Columns were reviewed and the STATUS and SPECIAL_CONSIDERATIONS columns were dropped as well as increasing the number of neurons and layers.  Other activations were tried such as tanh, but the range that model produced went from 40% to 68% accuracy.  The linear activation produced the worst accuracy, around 28%.  The relu activation at the early layers and sigmoid activation at the latter layers gave the best results.  
![Pic 2](https://github.com/zhangkevq/Neural_Network_Charity_Analysis/blob/main/images/model_performance.png)   
![Pic 3](https://github.com/zhangkevq/Neural_Network_Charity_Analysis/blob/main/images/model_performance_more_layers.png)    

### Summary:   
The relu and sigmoid activations yielded a 72.7% accuracy, which is the best the model could produce using various number of neurons and layers.  The next step should be to try the random forest classifier as it is less influenced by outliers.  
