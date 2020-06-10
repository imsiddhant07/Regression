import turicereate

sales = turicerate.SFrame('home_data.sframe')

import numpy as np

#----------------------------------------------------------------------------------------------------
#Some useful functions

def get_numpy_data(data_sframe,features,output):
	data_sframe['constant'] = 1
	features = ['constant'] + features
	features_sframe = data_sframe[features]
	feature_matrix = features_sframe.to_numpy()
	output_sarray = data_sframe[output]
	output_array = output_sarray.to_numpy()
	return(feature_matrix,output_array)

def predict_output(feature_matrix,weights):
	predictions = np.dot(feature_matrix,weights)
	return(predictions)

#----------------------------------------------------------------------------------------------------
#Compute the Derivative

def feature_derivative_ridge(errors,feature,weight,l2_penalty,feature_is_constant):
	if feature_is_constant == True :
		derivative = 2*np.dot(errors,feature)
	else :
		derivative = 2*np.dot(errors,feature) + 2*l2_penalty*weight
	return derivative

#To test your feature derivative
(example_features,example_output) = get_numpy_data(sales,['sqft_living'],'price')
my_weights = np.array([1.0,10.0])
test_predictions = predict_output(example_features,my_weights)
errors = test_predictions - example_output

print(feature_derivative_ridge(errors,example_features[:,1],my_weights[1],1,False))
print(np.sum(errors*example_features[:,1])*2+(2*1*my_weights[1]))
print('')

print(feature_derivative_ridge(errors,example_features[:,0],my_weights[0],1,True))
print(np.sum(errors*example_features[:,0])*2)
#Same as
print(np.sum(errors)*2)


def ridge_regresion_gradient_descent(feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations):
	print('Starting Gradient Descent with l2_penalty :',l2_penalty)
	weights = np.array(initial_weights)
	while max_iteration > 0:
		predictions = predit_output(feature_matrix,weights)
		errors = predictions - output
		for i in range(len(weights)):
			if i==0:
				feature_is_constant = True
			else :
				feature_is_constant =False
			derivative =feature_derivative_ridge(errors,feature_matrix[:,i],weights[i],l2_penalty,feature_is_constant)
			weights[i] -= (step_size*derivative)
		iterations -= 1
	return weights

simple_features = ['sqft_living']
my_output = 'price'
train_data,test_data = sales.random_split(0.8,seed=0)
(simple_feature_matrix,output) = get_numpy_data(train_data,simple_features,my_output)
(simple_test_feature_matrix,test_output) = get_numpy_data(test_data,simple_features,my_output)
initial_weights = np.array([0.0,0.0])
step_size = 1e-12
max_iterations = 1000

#l2_penalty = 0
ridged_weights_1 =  ridge_regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,0,max_iterations)
ridged_weights_1[0]
#-0.16311350055935017
ridged_weights_1[1]
#263.0243689065867


#l2_penalty = 1e11
ridged_weights_2 =  ridge_regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,1e11,max_iterations)
ridged_weights_2[0]
#9.767303834273747
ridged_weights_2[1]
#124.5721756462914

plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, ridged_weights_1),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, ridged_weights_2),'r-')
plt.show()

def compute_rss(simple_test_feature_matrix,test_output,weights):
	predictions = predict_output(simple_test_feature_matrix,weights)
	residuals = predictions - test_output
	rss = sum(residuals*residuals)
	return(rss)


rss_initial = compute_rss(simple_test_feature_matrix,test_output,initial_weights)
print(rss_initial)
#1784273282524564.0

rss_ridged_weights_1 = compute_rss(simple_test_feature_matrix,test_output,ridged_weights_1)
print(rss_ridged_weights_1)
#275723634597546.25
#Min RSS



rss_ridged_weights_2 = compute_rss(simple_test_feature_matrix,test_output,ridged_weights_2)
print(rss_ridged_weights_2)
#694642100913952.2

#-------------------------------------------------------------------------------------------------------
#Model with 2-features

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000

#l2_penalty = 0
ridged_weights_1 =  ridge_regression_gradient_descent(feature_matrix,output,initial_weights,step_size,0,max_iterations)
ridged_weights_1[0]
#-0.35743481889314127
ridged_weights_1[1]
#243.05416889988507
ridged_weights_1[2]
#22.414815939734385

#l2_penalty = 1e11
ridged_weights_2 =  ridge_regression_gradient_descent(feature_matrix,output,initial_weights,step_size,1e11,max_iterations)
ridged_weights_2[0]
#6.742965800693212
ridged_weights_2[1]
#91.48927361113033
ridged_weights_2[2]
#78.43658768266378


rss_initial = compute_rss(test_feature_matrix,test_output,initial_weights)
print(rss_initial)


rss_ridged_weights_1 = compute_rss(test_feature_matrix,test_output,ridged_weights_1)
print(rss_ridged_weights_1)




rss_ridged_weights_2 = compute_rss(test_feature_matrix,test_output,ridged_weights_2)
print(rss_ridged_weights_2)

