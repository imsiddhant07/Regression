import turicreate

#-----------------------------------------------------------------------------------------------
#Next we're going to write a polynomial function that takes an SArray and a maximal degree and returns an SFrame with columns containing the SArray to all the powers up to the maximal degree. The easiest way to apply a power to an SArray is to use the .apply() and lambda x: functions. For example to take the example array and compute the third power we can do as follows: 

tmp = turicreate.SArray([1.0,2.0,3.0])
tmp_cubed = tmp.apply(lambda x:x**3)
print(tmp)
print(tmp_cubed)

#We can create an empty SFrame using turicreate.SFrame() and then add any columns to it with ex_sframe['column_name'] = value.
ex_sframe = turicreate.SFrame()
ex_sframe['power1'] = tmp
print(ex_sframe)

#----------------------------------------------------------------------------------------------------
#Polynomial_sframe Function

def polynomial_sframe(feature,degree):
	poly_sframe = turicreate.SFrame()
	poly_sframe['power_1'] = feature
	if degree > 1:
		for power in range(2,degree+1):
			name = 'power_'+str(power)
			poly_sframe[name] = poly_sframe['power_1'].apply(lambda x:x**power)
	return poly_sframe

print(polynomial_sframe(tmp,3))

#----------------------------------------------------------------------------------------------------
# Visualizing polynomial regression

sales =turicreate.SFrame('home_data.sframe')

sales = sales.sort(['sqft_living','price'])


#NOTE: for all the models in this notebook use validation_set = None to ensure that all results are consistent across users.
#Degree-1
poly1_data = polynomial_sframe(sales['sqft_living'],1)
poly1_data['price'] = sales['price']

model1 = turicreate.linear_regression.create(poly1_data,target='price',features=['power_1'],validation_set=None,verbose=True)
model1.coefficients

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',poly1_data['power_1'],model1.predict(poly1_data),'-')
plt.show()


#Degree-2
poly2_data = polynomial_sframe(sales['sqft_living'],2)
my_features = poly2_data.column_names()
poly2_data['price']=sales['price']
model2 = turicreate.linear_regression.create(poly2_data,target='price',features=my_features,validation_set=None,verbose=True)
model2.coefficients
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',poly2_data['power_1'],model2.predict(poly2_data),'-')
plt.show()

#Degree-3
poly3_data = polynomial_sframe(sales['sqft_living'],3)
my_features = poly3_data.column_names()
poly3_data['price']=sales['price']
model3 = turicreate.linear_regression.create(poly3_data,target='price',features=my_features,validation_set=None,verbose=True)
model3.coefficients
plt.plot(poly3_data['power_1'],poly3_data['price'],'.',poly3_data['power_1'],model3.predict(poly3_data),'-')
plt.show()

#Degree-15
poly15_data = polynomial_sframe(sales['sqft_living'],15)
my_features = poly15_data.column_names()
poly15_data['price']=sales['price']
model15 = turicreate.linear_regression.create(poly15_data,target='price',features=my_features,validation_set=None,verbose=True)
model15.coefficients
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',poly15_data['power_1'],model15.predict(poly15_data),'-')
plt.show()

#----------------------------------------------------------------------------------------------
# Changing the data and re-learning

big_set1,big_set2 = sales.random_split(0.5,seed=0)
set_1,set_2 = big_set1.random_split(0.5,seed=0)
set_3,set_4 = big_set2.random_split(0.5,seed=0)

#Model for set_1
poly15_data1 = polynomial_sframe(set_1['sqft_living'],15)
my_features = poly15_data1.column_names()
poly15_data1['price'] = set_1['price']
model_151 = turicreate.linear_regression.create(poly15_data1,target='price',features=my_features,validation_set=None,verbose=True)
model_151.coefficients
plt.plot(poly15_data1['power_1'],poly15_data1['price'],'.',poly15_data1['power_1'],model_151.predict(poly15_data1))
plt.show()
#positive for power_15

#Model for set_2
poly15_data2 = polynomial_sframe(set_2['sqft_living'],15)
my_features = poly15_data2.column_names()
poly15_data2['price'] = set_2['price']
model_152 = turicreate.linear_regression.create(poly15_data2,target='price',features=my_features,validation_set=None,verbose=True)
model_152.coefficients
plt.plot(poly15_data2['power_1'],poly15_data2['price'],'.',poly15_data2['power_1'],model_152.predict(poly15_data2))
plt.show()
#positive for power_15

#Model for set_3
poly15_data3 = polynomial_sframe(set_3['sqft_living'],15)
my_features = poly15_data3.column_names()
poly15_data3['price'] = set_3['price']
model_153 = turicreate.linear_regression.create(poly15_data3,target='price',features=my_features,validation_set=None,verbose=True)
model_153.coefficients
plt.plot(poly15_data3['power_1'],poly15_data3['price'],'.',poly15_data3['power_1'],model_153.predict(poly15_data3))
plt.show()
#positive for power_15

#Model for set_4
poly15_data4 = polynomial_sframe(set_4['sqft_living'],15)
my_features = poly15_data4.column_names()
poly15_data4['price'] = set_4['price']
model_154 = turicreate.linear_regression.create(poly15_data4,target='price',features=my_features,validation_set=None,verbose=True)
model_154.coefficients
plt.plot(poly15_data4['power_1'],poly15_data4['price'],'.',poly15_data4['power_1'],model_154.predict(poly15_data4))
plt.show()
#negative for power_15

#-----------------------------------------------------------------------------------------------
# Selecting a Polynomial Degree
#Split our sales data into 2 sets: training_and_validation and testing. Use random_split(0.9,seed=1).
#Further split our training data into two sets: training and validation. Use random_split(0.5,seed=1).
#Again, we set seed=1 to obtain consistent results for different users

training_and_validation,test_data = sales.random_split(0.9,seed=1)
train_data,validation_data = training_and_validation.random_split(0.5,seed=1)

validation_rss=[]
for degree in range(1,16):
	poly_data = polynomial_sframe(train_data['sqft_living'],degree)
	my_features = poly_data.column_names()
	poly_data['price'] = train_data['price']
	model = turicreate.linear_regression.create(train_data,target='price',features=my_features,validation_set=None,verbose=False)
	validation_data_poly = polynomial_sframe(validation_data['sqft_living'],degree)
	predictions = model.predict(validation_data_poly)
	residuals = predictions-validation_data['price']	
	rs = residuals*residuals
	rss = rs.sum()
	validation_rss.append(rss)

test_rss=[]
for degree in range(1,16):
	poly_data = polynomial_sframe(train_data['sqft_living'],degree)
	my_features = poly_data.column_names()
	poly_data['price'] = train_data['price']
	model = turicreate.linear_regression.create(train_data,target='price',features=my_features,validation_set=None,verbose=False)
	test_data_poly = polynomial_sframe(test_data['sqft_living'],degree)
	predictions = model.predict(test_data_poly)
	residuals = predictions-test_data['price']	
	rs = residuals*residuals
	rss = rs.sum()
	test_rss.append(rss)










































































 
