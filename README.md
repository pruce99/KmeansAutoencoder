Intro To Machine Learning – Assignment
UB PERSON NUMBER: 50425014
UB IT NAME: PARAVAMU
TASK: The task of this project is to perform classification using machine learning. It is for 
a two-class problem. The task is to classify whether a patient has diabetes (class 1) or 
not (class 0), based on the diagnostic measurements provided in the dataset, using 
logistic regression and neural network as the classifier. The dataset in use is the Pima 
Indians Diabetes Database(diabetes.csv). The code should be written in Python.
STEPS INVOLVED: 
1.) IMPORTING NECESSARY LIBRARIES
ð Importing numpy and pandas for data processing 
ð Importing matplotlib to plot loss and accuracy change graph
2.) IMPORTING THE DATASET
ð We use Pandas to read our csv and save it in the form of pandas dataframe
ð Dataset.head() lets us visualize the first 5 rows of the dataset
ð This is usually done to see if the dataset is imported properly
ð Also to get more information about the dataset we use dataset.describe() to 
get standard deviation, max, min, count and mean of the dataset
3.) PROCESSING THE DATASET
ð Our first objective is to split the given dataset to train, test and validate 
ð We can use numpy split to do this as shown below 
ð Our next objective is to save the feature columns to train_X, test_X and 
Validation_X
ð Then we save the outcome column to train_Y, test_Y, and validation_Y
ð To make sure everything is right we visualize the shape of train, test and 
validation set
4.) NORMALIZING THE DATA
ð We normalize the data so the gradient can reach the global minima sooner
ð Working with big values will make it take a long time and oscillate back and 
forth and will take more time to reach the global minima 
ð There are many ways to normalize the data, the Zscore approach is used 
below to scale the data between 0 to 1
5.) DEFINING THE HYPOTHESIS FUNCTION OR THE SIGMOID FUNCTION
ð The hypothesis function of logistic regression is a sigmoid function 
ð The sigmoid function allows to limit the cost function between 0 to 1
ð We also use the sigmoid function to map the predicted values to probabilities
ð The formula is 1/(1+e^(-z)) where z = (train_X x weight) + bias
6.) DEFINING THE COST FUNCTION
ð In logistic regression there exists two cost functions which we later combine 
to one function.
ð When y = 1 we say the cost function is -log (predicted y) where predicted y is 
the output from our sigmoid hypothesis function
ð When y = 0 we say the cost function is -log (1 - predicted y) 
ð The reason why we have two cost functions is because if we plot -log (x),
when our hypothesis approach 0 the cost function approaches infinity
ð Similarly, the vice versa happens when the cost function is -log (1 – predicted
y)
ð We can combine the cost functions into one 
ð The combined cost function is -ylog (predicted y) – (1-y) x log (1 - predicted y)
ð When y = 0 the first half is negated and when y = 1 the second half is negated 
ð For m training data we find the cost.
ð Adding a eps to avoid nan value
7.) DEFINING THE GRADIENT DESCENT FUNCTION
ð We find the gradient with respect to weight 
ð We then find the gradient with respect to bias
ð We then update the new weight and new bias by multiplying it with the 
learning rate and subtracting from the old weight and old bias
ð This runs epoch number of times
8.) TRAINING THE DATA
ð This function calls all the defined functions and lets everything fall into place
ð We initial the weights by randomizing the weights and initializing the bias to 
be 0
ð We use the normalized X values; we find the hypothesis by using the sigmoid 
function 
ð With the help of the sigmoid outputs, we will be able to find the gradients and 
update our new weights and bias
ð This will run based on n number of epochs
ð We also call the accuracy function to track the accuracy
ð We return the weight, bias and losses
ð The training can differ based on the learning rate (lr) we pick and the number 
of epochs we pick 
ð We tune this until we get better accuracy and a good loss curve
9.) PLOTING THE LOSS AND ACCURACY GRAPH
ð We initialize a lost list and accuracy list to track the loss and accuracy
ð These values can be used to visualize the growth of loss and accuracy 
ð The accuracy increases as the training happens. 
10.) PREDICTION AND CHECKING ACCURACY 
ð For prediction we use the updated weights and bias which we got through our 
training
ð We use the trained weights and bias in our sigmoid function for prediction
ð We normalize X before we pass it through the sigmoid function
ð If the predicted value is above 0.5 then the patient has diabetes and if the 
predicted value is below 0.5 then patient doesn’t have diabetes
ð We use the accuracy function to see if the predictions made is right or wrong 
ð The accuracy was found to be 
Part 2: Applying Neural Networks
1.) Importing Necessary Libraries
ð We import TensorFlow which handles data sets that are arrayed as 
computational nodes in graph form.
2.) Defining the layers of the neural network 
ð The first layer is the input layer which consists of the features
ð In our case the feature vector has 8 features 
ð The second layer is a dense layer which is connected deeply 
ð The neuron in the dense layer receives input from the previous layer 
ð In background the dense layer performs matrix multiplication.
ð The dense layer is also given Relu activation as parameter 
ð The dense layer can take multiple parameters, the number of units and the 
activation function is used in our program
Relu Activation
o Rectified linear Unit (ReLu) is a activation function which does a simple 
function.
o If the value is positive, it leaves it as it is.
o If the value is negative, it replaces it with 0.
o It just does Max (0, number)
o Relu activation overcomes the vanishing gradient problem.
o It is the best activation function compared to sigmoid and tangent when 
it comes to a neural network with multiple layers.
ð The third layer is also a hidden layer with units and relu activation.
ð The final and last layer is the output layer. Since we are doing binary 
classification where there can be only one output at a time, we use one unit 
and a sigmoid activation function.
3.) Compiling and Fitting the Model.
ð Model(): groups layers into an object with training and inference features.
ð We then compile the model with the binary entropy loss function. 
ð We use the Adam optimizer with a desired learning rate and specify the 
metrics we want to display.
ð The Adam optimizer is good with sparse data, it uses an adaptive learning 
rate.
ð We don’t have to worry about the learning rate unlike Stochastic gradient 
descent.
ð It’s the best among the adaptive optimizers.
ð We call our function and store it in an object
ð To get the summary about output shape and params of each layer we use the 
summary() function 
ð Then we use the fit function using the train_X and train_Y data for training 
and use validation_X and Validation_y for validating purposes.
ð We mention the number of epochs and the batch size also.
ð We can see the decrease in loss and the increase in accuracy
4.) Plotting Loss and Accuracy 
ð Using our plot function, we plot the fit model’s loss and validation loss.
ð From the graph below there’s a steady decrease of the loss and the validation 
loss.
ð Similarly, we must plot the accuracy and the validation accuracy.
ð From the graph we can see a steady increase in Accuracy.
5.) Model Evaluation
ð We evaluate the model using the train, test and validation data.
ð It gives the loss and accuracy of the model.
Part 3 Using Different Regularizes
1.) Using Dropout Regularizer
ð Dropout Regularizer sets input units to zero randomly during the training time.
ð It’s a simple way to avoid the overfitting problem.
ð We randomly drop the units along with their connections. 
ð Co – adapting a lot is also a serious issue, dropout Regularizer takes care of 
that.
ð We follow the same code as we did above, but we add two dropout layers 
within our dense layers.
ð We pass a dropout percentage of 40.
ð We plot the loss of our new model with dropout regularization.
ð We plot the accuracy of our new model with dropout regularization.
ð Results from dropout Regularizer evaluation.
2.) Using L1 Regularizer
ð L1 Regularizer is also called as Lasso Regression.
ð There is also a L2 Regularizer, but the difference is the penalty term.
ð L1 Regularizer adds lambda penalty value (absolute value of magnitude) with 
changes in signs as same as the weight tensor. 
ð We use the L1 Regularizer in our dense layer as shown in the below image.
ð It reduces the value of the less important features.
ð This works well when we have a huge number of features.
ð Plotting the loss and accuracy of the L1 regularized model.
ð Results from L1 Regularizer evaluation.
3.) Using L2 Regularizer
ð L2 Regularizer is also called as Ridge Regression
ð This adds the squared magnitude as the penalty term.
ð We must choose lambda carefully cause big values may lead to underfitting.
ð We use the L2 Regularizer as how we used the L1 Regularizer.
ð Multiply twice the lambda value with weights matrix.
ð Plotting the loss and the accuracy of the L2 Regularized model.
ð Results from L2 Regularizer evaluation.
