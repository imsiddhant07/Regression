import turicreate

sales = turicreate.SFrame('home_data.sframe')

sales['floors'] = sales['floors'].astype(float).astype(int)

import numpy as np


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

#Normalize features
X = np.array([[3.0,5.0,8.0],[4.0,12.0,15.0]])
print(X)

norms = np.linalg.norm(X,axis=0)
print(norms)

print(X/norms)

def normalize_features(feature_matrix):
	norms = np.linalg.norm(feature_matrix,axis=0)
	normalize_features = feature_matrix/norms
	return(normalize_features,norms)

features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print(features)
# should print
# [[ 0.6  0.6  0.6]
#  [ 0.8  0.8  0.8]]
print(norms)
# should print
# [5.  10.  15.]

#-------------------------------------------------------------------------------------------------
# Implementing Coordinate Descent with normalized features

simple_features = ['sqft_living','bedrooms']
my_output = 'price'
(simple_feature_matrix,output) = get_numpy_data(sales,simple_features,my_output)
simple_feature_matrix,norm = normalize_features(simple_feature_matrix)
weights = np.array([1.0,4.0,1.0])
predictions = predict_output(simple_feature_matrix,weights)

ro = [0 for i in range((simple_feature_matrix.shape)[1])]

for i in range((simple_feature_matrix.shape)[1]):
	ro[i] = (simple_feature_matrix[:,i]*(output-predictions+weights[i]*simple_feature_matrix[:,i])).sum()
print(ro)
#[79400300.03492916, 87939470.77299108, 80966698.67596565]


#we have ro[0], ro[1], ro[2]
#For W1 to be zero, we need ro[1] in [-lambda/2, lambda/2]
#We have -lambda/2 <= ro[1] <= lambda/2
#This translates to lambda >= -2ro[1] and lambda >= 2ro[1]
#For both conditions to be satisfied, lambda >= 2ro[1] = 1.75e8
#Similarly for W2. lambda >= 2ro[2] = 1.62e8
#So, w[i] = 0 if lambda >= 2 * abs(ro[i])

print(2*ro[1])
#175878941.54598215
print(2*ro[2])
#161933397.3519313

def in_l1range(value, penalty):
    return ( (value >= -penalty/2.) and (value <= penalty/2.) )

for l1_penalty in [1.4e8, 1.64e8, 1.73e8, 1.9e8, 2.3e8]:
    print(in_l1range(ro[1], l1_penalty), in_l1range(ro[2], l1_penalty))




## Single Coordinate Descent Step

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix,weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = sum((feature_matrix[:,i])*(output-predictions+weights[i]*feature_matrix[:,i]))
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.
    return new_weight_i

# should print 0.425558846691
import math
print(lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]),np.array([1., 1.]), np.array([1., 4.]), 0.1))

### Cyclical coordinate descent 
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    D = feature_matrix.shape[1]
    weights = np.array(initial_weights)
    change = np.array(initial_weights) * 0.0
    converged = False

    while not converged:

    # Evaluate over all features
        for idx in range(D):
#             print 'Feature: ' + str(idx)
            # new weight for feature
            new_weight = lasso_coordinate_descent_step(idx, feature_matrix,
                                                       output, weights,
                                                       l1_penalty)
            # compute change in weight for feature
            change[idx] = np.abs(new_weight - weights[idx])
#             print '  -> old weight: ' + str(weights[idx]) + ', new weight: ' + str(new_weight)
#             print '  -> abs change (new - old): ' + str(change[idx])
#             print '  >> old weights: ', weights

            # assign new weight
            weights[idx] = new_weight
#             print '  >> new weights: ', weights
        # maximum change in weight, after all changes have been computed
        max_change = max(change)
#         print '  ** max change: ' + str(max_change)
#         print '--------------------------------------------------'
        if max_change < tolerance:
            converged = True
    return weights



simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) 
# normalize features

weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
print(weights)
#[ 21624998.36636292  63157246.78545421  0.0]



prediction =  predict_output(normalized_simple_feature_matrix, weights)
RSS = np.dot(output-prediction, output-prediction)
print('RSS for normalized dataset = ', RSS)
#RSS for normalized dataset =  1.63049248148e+15


#Evaluating LASSO fit with more features
train_data,test_data = sales.random_split(0.8,seed=0)


all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']

my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, all_features, my_output)
normalized_feature_matrix, norms = normalize_features(feature_matrix)

initial_weights = np.zeros(len(all_features) + 1)
l1_penalty = 1e7
tolerance = 1.0

weights1e7 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output,
                                               initial_weights, l1_penalty, tolerance)

print(weights1e7)
#[ 24429600.60933314         0.                 0.          48389174.35227978
#         0.                 0.           3317511.16271982   7329961.9848964
#         0.                 0.                 0.                 0.
#         0.                 0.        ]

feature_list = ['constant'] + all_features
print(feature_list)
#['constant', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']




feature_weights1e7 = dict(zip(feature_list, weights1e7))
for k,v in feature_weights1e7.iteritems():
    if v != 0.0:
        print(k, v)
#sqft_living 48389174.3523
#waterfront 3317511.16272
#constant 24429600.6093
#view 7329961.9849

l1_penalty=1e8
tolerance = 1.0
weights1e8 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output,
                                               initial_weights, l1_penalty, tolerance)

print(weights1e8)
#[ 71114625.75280938         0.                 0.                 0.
#         0.                 0.                 0.                 0.
#         0.                 0.                 0.                 0.
#         0.                 0.        ]


feature_weights1e8 = dict(zip(feature_list, weights1e8))
for k,v in feature_weights1e8.iteritems():
    if v != 0.0:
        print(k, v)
#constant 71114625.75280938 


l1_penalty=1e4
tolerance=5e5
weights1e4 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output,
                                               initial_weights, l1_penalty, tolerance)
print(weights1e4)
#[ 77779073.91265225 -22884012.25023361  15348487.08089996
 # 92166869.69883074  -2139328.0824278   -8818455.54409492
 #  6494209.73310655   7065162.05053198   4119079.21006765
 # 18436483.52618776 -14566678.54514342  -5528348.75179426
 #-83591746.20730537   2784276.46012858]

feature_weights1e4 = dict(zip(feature_list, weights1e4))
for k,v in feature_weights1e4.iteritems():
    if v != 0.0:
        print(k, v)

#bathrooms 15348487.0809
#sqft_above -14566678.5451
#grade 18436483.5262
#yr_renovated 2784276.46013
#bedrooms -22884012.2502
#sqft_living 92166869.6988
#floors -8818455.54409
#condition 4119079.21007
#waterfront 6494209.73311
#constant 77779073.9127
#sqft_basement -5528348.75179
#yr_built -83591746.2073
#sqft_lot -2139328.08243
#view 7065162.05053

#Rescaling learned weights
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, all_features, my_output)
normalized_feature_matrix, norms = normalize_features(feature_matrix)
normalized_weights1e7 = weights1e7 / norms
normalized_weights1e8 = weights1e8 / norms
normalized_weights1e4 = weights1e4 / norms
print(normalized_weights1e7[3])

#Evaluating each of the learned models on the test data
(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')

prediction =  predict_output(test_feature_matrix, normalized_weights1e7)
RSS = np.dot(test_output-prediction, test_output-prediction)
print('RSS for model with weights1e7 = ', RSS)
#RSS for model with weights1e7 =  2.75962079909e+14

prediction =  predict_output(test_feature_matrix, normalized_weights1e8)
RSS = np.dot(test_output-prediction, test_output-prediction)
print('RSS for model with weights1e8 = ', RSS)
#RSS for model with weights1e8 =  5.37166150034e+14

prediction =  predict_output(test_feature_matrix, normalized_weights1e4)
RSS = np.dot(test_output-prediction, test_output-prediction)
print('RSS for model with weights1e4 = ', RSS)
#RSS for model with weights1e4 =  2.2778100476e+14






































































