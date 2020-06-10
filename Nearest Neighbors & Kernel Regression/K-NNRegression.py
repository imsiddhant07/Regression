import turicreate

sales = turicreate.SFrame('home_data_small.sframe')

import numpy as np

def get_numpy_data(data_sframe,features,output):
	data_sframe['constant'] = 1
	features = ['constant'] + features
	features_sframe = data_sframe[features]
	feature_matrix = features_sframe.to_numpy()
	output_sarray = data_sframe[output]
	output_array = output_sarray.to_numpy()
	return(feature_matrix,output_array)

def normalize_features(feature_matrix):
	norms = np.linalg.norm(feature_matrix,axis=0)
	normalize_features = feature_matrix/norms
	return(normalize_features,norms)

# initial train/test split
(train_and_validation, test) = sales.random_split(0.8, seed=1) 
# split training set into training and validation sets
(train, validation) = train_and_validation.random_split(0.8, seed=1) 

#Extract features and normalize them
feature_list = ['bedrooms',  
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
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train)
 # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms


































