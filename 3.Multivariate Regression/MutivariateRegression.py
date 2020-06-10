import turicreate

#Load the house sales data
sales = turicerate.SFrame('home_data.sframe')

#Split data into train and test data
train_data,test_data = sales.random_split(0.8,seed=0)


#Learning a multiple regression model 
example_features = ['sqft_living','bedrooms','bathrooms']
example_model = turicreate.linear_regression.create(train_data,target='price',features=example_features,validation_set=None,verbose=False)
example_weight_summary = example_model.coefficients
print(example_weight_summary)

#Making predictions
example_predictions = example_model.predict(train_data)
print(example_predictions[0])

#Compute RSS
def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
	predictions = model.predict(data)
    # Then compute the residuals/errors
	residual = predictions - outcome
    # Then square and add them up
	residual_sq = residual*residual
	RSS = residual_sq.sum()
    return(RSS)    
#Test your function by computing the RSS on TEST data for the example model:
rss_example_train = get_residual_sum_of_squares(example_model,test_data,test_data['price'])
print(rss_example_train)			#273761538330193.0

#Create some new features
from math import log
#Next create the following 4 new features as column in both TEST and TRAIN data:
#* bedrooms_squared = bedrooms*bedrooms
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x:x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x:x**2)
#* bed_bath_rooms = bedrooms*bathrooms
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']
#* log_sqft_living = log(sqft_living)
train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x:log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x:log(x))
#* lat_plus_long = lat + long 
train_data['lat_plus_long'] = train_data['lat']+train_data['long']
test_data['lat_plus_long'] = test_data['lat']+test_data['long']

test_data['bedrooms_squared'].mean()
#12.446677701584301
test_data['bed_bath_rooms'].mean()
#7.503901631591394
test_data['log_sqft_living'].mean()
#7.550274679645938
test_data['lat_plus_long'].mean()
#-74.65333497217307



# Learning Multiple Models
model_1_features = ['sqft_living','bedrooms','bathrooms','lat','long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared','log_sqft_living','lat_plus_long']

model1 = turicreate.linear_regression.create(train_data,target='price',features=model_1_features,validation_set=None,verbose=True)
model1_coefficients = model1.coefficients
print(model1_coefficients)
#+-------------+-------+---------------------+--------------------+
#|     name    | index |        value        |       stderr       |
#+-------------+-------+---------------------+--------------------+
#| (intercept) |  None |  -56140675.74114427 | 1649985.420135553  |
#| sqft_living |  None |  310.26332577692136 | 3.1888296040737765 |
#|   bedrooms  |  None |  -59577.11606759667 | 2487.2797732245012 |
#|  bathrooms  |  None |  13811.840541653264 | 3593.5421329670735 |
#|     lat     |  None |  629865.7894714845  | 13120.710032363884 |
#|     long    |  None | -214790.28516471002 | 13284.285159576597 |
#+-------------+-------+---------------------+--------------------+
#[6 rows x 4 columns]

model2 = turicreate.linear_regression.create(train_data,target='price',features=model_2_features,validation_set=None,verbose=True)
model2_coefficients = model2.coefficients
print(model2_coefficients)
#+----------------+-------+---------------------+--------------------+
#|      name      | index |        value        |       stderr       |
#+----------------+-------+---------------------+--------------------+
#|  (intercept)   |  None |  -54410676.1071702  | 1650405.1652726454 |
#|  sqft_living   |  None |  304.44929805407946 |  3.20217535637094  |
#|    bedrooms    |  None | -116366.04322451768 | 4805.5496654858225 |
#|   bathrooms    |  None |  -77972.33050970349 | 7565.059910947983  |
#|      lat       |  None |  625433.8349445503  | 13058.353097300462 |
#|      long      |  None | -203958.60289731968 | 13268.128370009661 |
#| bed_bath_rooms |  None |  26961.624907583264 | 1956.3656155588428 |
#+----------------+-------+---------------------+--------------------+
#[7 rows x 4 columns]

model3 = turicreate.linear_regression.create(train_data,target='price',features=model_3_features,validation_set=None,verbose=True)
model3_coefficients = model3.coefficients
print(model3_coefficients)
#+------------------+-------+---------------------+--------------------+
#|       name       | index |        value        |       stderr       |
#+------------------+-------+---------------------+--------------------+
#|   (intercept)    |  None |  -52974974.06892153 | 1615194.942821453  |
#|   sqft_living    |  None |  529.1964205687523  | 7.699134985078978  |
#|     bedrooms     |  None |  28948.527746351134 | 9395.728891110177  |
#|    bathrooms     |  None |  65661.20723969836  | 10795.338070247015 |
#|       lat        |  None |  704762.1484430869  |        nan         |
#|       long       |  None | -137780.02000717327 |        nan         |
#|  bed_bath_rooms  |  None |  -8478.364107167803 | 2858.9539125640354 |
#| bedrooms_squared |  None |  -6072.384661904947 | 1494.9704277794906 |
#| log_sqft_living  |  None |  -563467.7842801767 | 17567.823081204006 |
#|  lat_plus_long   |  None |  -83217.19791002883 |        nan         |
#+------------------+-------+---------------------+--------------------+
#[10 rows x 4 columns]


#Residual sum of squares on train data
rss_train_model_1 = get_residual_sum_of_squares(model1,train_data,train_data['price'])
print(rss_model_1)
#971328233545434.4

rss_train_model_2 = get_residual_sum_of_squares(model2,train_data,train_data['price'])
print(rss_model_2)
#961592067859822.1

rss_train_model_3 = get_residual_sum_of_squares(model3,train_data,train_data['price'])
print(rss_model_3)
#905276314551640.9


#Residual sum of squares on test data
rss_test_model_1 = get_residual_sum_of_squares(model1,test_data,test_data['price'])
print(rss_test_model_1)
#226568089093160.56

rss_test_model_2 = get_residual_sum_of_squares(model2,test_data,test_data['price'])
print(rss_test_model_2)
#224368799994313.0

rss_test_model_3 = get_residual_sum_of_squares(model3,test_data,test_data['price'])
print(rss_test_model_3)
#251829318963157.28





















