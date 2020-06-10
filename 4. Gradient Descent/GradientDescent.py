import turicreate 
import numpy as np

#------------------------------------------------------------------------------------------------

sales = turicreate.SFrame('home_data.sframe')

#------------------------------------------------------------------------------------------------

def get_numpy_data(data_sframe,features,output):
	data_sframe['constant'] = 1
	features = ['constant'] + features
	features_sframe = data_sframe[features]
	feature_matrix = features_sframe.to_numpy()
	output_sarray = data_sframe[output]
	output_array = output_sarray.to_numpy()
	return(feature_matrix,output_array)


(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
# the [] around 'sqft_living' makes it a list
print(example_features[0,:])
# this accesses the first row of the data the ':' indicates 'all columns'
print(example_output[0])			 # and the corresponding output

#----------------------------------------------------------------------------------------------
# Predicting output given regression weights

my_weights = np.array([1,1])
my_features = example_features[0,]
predicted_value = np.dot(my_features,my_weights)
print(predicted_value)

#Function to predict output
def predict_output(feature_matrix,weights):
	predictions = np.dot(feature_matrix,weights)
	return(predictions)

test_predictions = predict_output(example_features,my_weights)
print(test_predictions[0])
#1181.0
print(test_predictions[1])
#2571.0

#----------------------------------------------------------------------------------------------
#Computing the derivative

def feature_derivative(errors,feature):
	derivative = 2*np.dot(errors,feature)
	return(derivative)
 
(example_features,example_output) = get_numpy_data(sales,['sqft_living'],'price')
my_weights = np.array([0,0])
test_predictions = predict_output(example_features,my_weights)
errors = predictions - example_output
feature = example_features[:,0]
derivative = feature_derivative(errors,feature)
print(derivative)
print(-np.sum(example_output)*2)

#----------------------------------------------------------------------------------------------
#Gradient Descent
from math import sqrt

def regression_gradient_descent(feature_matrix,output,initial_weights,step_size,tolerance):
	converged = False
	weights = np.array(initial_weights)
	predictions = predict_output(feature_matrix,weights)
	errors = predictions - output
	gradient_sum_squares = 0
	for i in range(len(weights)):
		derivative = feature_derivative(errors,feature_matrix[:,i])
		gradient_sum_squares += (derivative**2)
		weights[i] -= (step_size*derivative)
	gradient_magnitude = sqrt(gradient_sum_squares)
	if gradient_magniude < tolerance:
		converged = True
	return(weights)

#----------------------------------------------------------------------------------------------
# Running the Gradient Descent as Simple Regression
train_data,test_data = sales.random_split(0.8,seed=0)
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix,output) = get_numpy_data(train_data,simple_features,my_output)
initial_weights = np.array([-47000.0,1.0])
step_size = 7e-12
tolerance 2.5e7

weights = regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,tolerance)

weights
#array([-46999.85779866,    354.86068685])

(test_simple_feature_matrix,test_output) = get_numpy_data(test_data,simple_features,my_output)

#Now compute your predictions using test_simple_feature_matrix and your weights from above.
test_predictions = predict_output(test_simple_feature_matrix,weights)

test_predictions[0]
#460450.9243946416
test_output[0]
#310000.0


residual = test_predictions - test_output
residual_squared = residual**2
RSS = residual_squared.sum()
print(RSS)
#389725351260830.5

#----------------------------------------------------------------------------------------------
#Running a Multiple Regression

model_features = ['sqft_living','sqft_living15']
my_output = 'price'
(feature_matrix,output) = get_numpy_data(train_data,model_features,my_output)
initial_weights = np.array([-100000.0,1.0,1.0])
step_size = 4e-12
tolerance = 1e9

model_weights = regression_gradient_descent(feature_matrix,output,initial_weights,step_size,tolerance)
model_weights

(test_feature_matrix,test_model_output) = get_numpy_data(test_data,model_features,my_output)

test_model_predictions = predict_output(test_feature_matrix,model_weights)

test_model_predictions[0]
#562125.8878434541

residual = test_model_predictions - test_model_output
re_sq = residual*residual
RSS = re_sq.sum()
print(RSS)
#463885093574075.94























































