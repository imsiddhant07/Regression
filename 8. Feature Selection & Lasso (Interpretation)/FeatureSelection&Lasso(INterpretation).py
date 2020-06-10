import turicreate 
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

sales = turicreate.SFrame('home_data.sframe')

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']


# Learn regression weights with L1 penalty
all_features = ['bedrooms', 'bedrooms_square',
                'bathrooms',
                'sqft_living', 'sqft_living_sqrt',
                'sqft_lot', 'sqft_lot_sqrt',
                'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 'yr_renovated']

model_all = turicreate.linear_regression.create(sales,target='price',features=all_features,validation_set=None,l2_penalty=0.0,l1_penalty=1e10)
model_all.coefficients.print_rows(18)
#+------------------+-------+--------------------+--------+
#|       name       | index |       value        | stderr |
#+------------------+-------+--------------------+--------+
#|   (intercept)    |  None | 274873.05595049576 |  None  |
#|     bedrooms     |  None |        0.0         |  None  |
#| bedrooms_square  |  None |        0.0         |  None  |
#|    bathrooms     |  None | 8468.531086910105  |  None  |
#|   sqft_living    |  None | 24.42072098244546  |  None  |
#| sqft_living_sqrt |  None | 350.06055338605546 |  None  |
#|     sqft_lot     |  None |        0.0         |  None  |
#|  sqft_lot_sqrt   |  None |        0.0         |  None  |
#|      floors      |  None |        0.0         |  None  |
#|  floors_square   |  None |        0.0         |  None  |
#|    waterfront    |  None |        0.0         |  None  |
#|       view       |  None |        0.0         |  None  |
#|    condition     |  None |        0.0         |  None  |
#|      grade       |  None | 842.0680348976231  |  None  |
#|    sqft_above    |  None | 20.024722417091112 |  None  |
#|  sqft_basement   |  None |        0.0         |  None  |
#|     yr_built     |  None |        0.0         |  None  |
#|   yr_renovated   |  None |        0.0         |  None  |
#+------------------+-------+--------------------+--------+
#[18 rows x 4 columns]


# Selecting an L1 penalty
# initial train/test split
(training_and_validation, testing) = sales.random_split(0.9,seed=1) 
# split training into train and validate
(training, validation) = training_and_validation.random_split(0.5, seed=1)

val_err_dict = {}
for l1_penalty in np.logspace(1,7,num=13):
	model = turicreate.linear_regression.create(training,target='price',features=all_features,l2_penalty=0,l1_penalty=l1_penalty,validation_set=None,verbose=False)
	predictions = model.predict(validation)
	residuals = predictions - validation['price']
	val_err = sum(residuals*residuals)
	print('For l1_penalty:'+str(l1_penalty)+' the validation error is:'+str(val_err))
	val_err_dict[l1_penalty]=val_err


print(val_err_dict)
#{10.0: 625766285142461.2, 31.622776601683793: 625766285362395.4, 100.0: 625766286057886.9, 316.22776601683796: 625766288257224.8, 1000.0: 625766295212185.9, 3162.2776601683795: 625766317206077.6, 10000.0: 625766386760661.5, 31622.776601683792: 625766606749281.1, 100000.0: 625767302791633.1, 316227.7660168379: 625769507643885.0, 1000000.0: 625776517727025.6, 3162277.6601683795: 625799062845466.8, 10000000.0: 625883719085424.4}

print(min(val_err_dict.items(),key=lambda x:x[1]))
#(10.0, 625766285142461.2)
	
l1_penalty = turicreate.SArray(val_err_dict.keys())
validation_error = turicreate.SArray(val_err_dict.values())
sf = turicreate.SFrame({'l1_penalty':l1_penalty,'validation_error':validation_error})
print(sf)

plt.plot(sf['l1_penalty'],sf['validation_error'],'k.')
plt.xscale('log')
plt.show()

model_l1_10 =  turicreate.linear_regression.create(training,target='price',features=all_features,l2_penalty=0,l1_penalty=10,validation_set=None,verbose=False)
>>> model_l1_10.coefficients.print_rows(18)
#+------------------+-------+----------------------+--------+
#|       name       | index |        value         | stderr |
#+------------------+-------+----------------------+--------+
#|   (intercept)    |  None |   18993.4272127706   |  None  |
#|     bedrooms     |  None |  7936.9676790313015  |  None  |
#| bedrooms_square  |  None |  936.9933681932994   |  None  |
#|    bathrooms     |  None |  25409.588934120668  |  None  |
#|   sqft_living    |  None |  39.11513637970764   |  None  |
#| sqft_living_sqrt |  None |  1124.650212807717   |  None  |
#|     sqft_lot     |  None | 0.003483618222989654 |  None  |
#|  sqft_lot_sqrt   |  None |  148.25839101140826  |  None  |
#|      floors      |  None |  21204.335466950117  |  None  |
#|  floors_square   |  None |  12915.524336072433  |  None  |
#|    waterfront    |  None |   601905.594545272   |  None  |
#|       view       |  None |  93312.85731187189   |  None  |
#|    condition     |  None |  6609.035712447216   |  None  |
#|      grade       |  None |  6206.939991880552   |  None  |
#|    sqft_above    |  None |  43.287053419335614  |  None  |
#|  sqft_basement   |  None |  122.3678275341193   |  None  |
#|     yr_built     |  None |  9.433635393724911   |  None  |
#|   yr_renovated   |  None |  56.072003448822365  |  None  |
#+------------------+-------+----------------------+--------+

#--------------------------------------------------------------------------------------------------
#Limiting the number of non-zero weights 

max_nonzeros = 7
l1_penalty_values = np.logspace(8,10,num=20)

def non_zero_counter(value):
	non_zero = 0
	for i in range(len(value)):
		if value[i] != 0:
			non_zero += 1
		else:
	return(non_zero)

for l1_penalty in np.logapace(8,10,num=20):
	model = turicreate.linear_regression.create(training,target='price',features=all_features,l2_penalty=0,l1_penalty=l1_penalty,validation_set=None,verbose=False)
	value = model.coefficients['value']
	non_zeros = non_zero_counter(value)
	print('Number of non-zero coefficients for l1_penalty:'+str(l1_penalty)+' is:'+str(non_zeros))





















