import turicreate

# Load house sales data
#Dataset is from house sales in King County, the region where the city of Seattle, WA is located.
sales = turicreate.SFrame('house_data.sframe')

# Split data into training and testing
train_data,test_data = sales.random_split(0.8,seed=0)


# Useful SFrame summary functions
#In order to make use of the closed form solution as well as take advantage of turi create's built in functions we will review some important ones. In particular:
#* Computing the sum of an SArray
#* Computing the arithmetic average (mean) of an SArray
#* multiplying SArrays by constants
#* multiplying SArrays by other SArrays

#Computing average price of houses - Method 1
prices = sales['price']
sum_price = prices.sum()
num_houses = len(prices)
avg_price1 = sum_price/num_houses
#Computing average price of houses - Method 2
avg_price2 = prices.mean()
print("Average price via method 1:",avg_price1)
print("Average price via method 2:",avg_price2)

# if we want to multiply every price by 0.5 it's a simple as:
half_prices = 0.5*prices
#Let's compute sum of squares of prices 
prices_squared = prices*prices
sum_prices_squared = prices_squared.sum()
print("Sum of squares for prices is",sum_prices_squared)



# Build a generic simple linear regression function 
def simple_linear_regression(input_feature,output):
    # compute the sum of input_feature and output
    sum_input_feature = input_feature.sum()  	#sum(xi)
	sum_output = output.sum()					#sum(yi)
    # compute the product of the output and the input_feature and its sum
    prod_input_output = input_feature*output				#(yi*xi)
	sum_prod_input_output = prod_input_output.sum()  		#(sum(yi*xi))
	prod_of_sum = sum_input_feature*sum_output              #(sum(yi)*sum(xi))
    # compute the squared value of the input_feature and its sum
    input_squared = input_feature*input_feature	            #xi*xi
	sum_input_squared = input_squared.sum()					#sum(xi*xi)
	prod_sum_input = sum_input_feature*sum_input_feature	#(sum(xi)*sum(xi))
	N = len(input_feature)
    # use the formula for the slope
    slope = ((sum_prod_input_output)-(prod_of_sum/N))/((sum_input_squared)-(prod_sum_input/N))
    # use the formula for the intercept
    intercept = (sum_output/N)-((slope*sum_input_feature)/N)
    return (intercept, slope)

test_feature = turicreate.SArray(range(5))
test_output = turicreate.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print("Intercept: " + str(test_intercept))
print("Slope: " + str(test_slope))

#Now that we know it works let's build a regression model for predicting price based on sqft_living. Rembember that we train on train_data!
(sqft_intercept,sqft_slope) = simple_linear_regression(train_data['sqft_living'],train_data['price'])
print("Sqft Intercept is:",sqft_intercept)
print("Sqft Slope is:",sqft_slope)



# Predicting Values
def get_regression_predictions(input_feature,intercept,slope):
	predicted_values = intercept + slope*input_features
	return predicted_values

my_house_sqft = 2650
estimated_price = get_regression_prediction(my_house_sqft,sqft_intercept,sqft_slope)
print("The estimated price for a house with {}sqft of area is {}$".format(my_house_sqft,estimated_price))
#The estimated price for a house with 2650sqft of area is 700074.8456294581$



# Residual Sum of Squares
def get_residual_sum_of_squares(input_feature,output,intercept,slope):
    # First get the predictions
	predictions = get_regression_predictions(input_feature,intercept,slope)
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
	residual = predictions-output
    # square the residuals and add them up
	residual_squared = residual*residual
	RSS = residual_squared.sum()
    return(RSS)

print(get_residual_sum_of_squares(test_feature,test_output,test_intercept,test_slope))

rss_price_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'],train_data['price'],sqft_intercept,sqft_slope)
print('The RSS of predicting Prices based on Square Feet is : ',rss_prices_on_sqft)
#The RSS of predicting Prices based on Square Feet is :  1201918356321966.8



# Predict the squarefeet given price
def inverse_regression_predictions(output,intercept,slope):
	estimated_feature = (output-intercept)/slope
	return estimated_value

my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price,sqft_intercept,sqft_slope)
print("The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet))
#The estimated squarefeet for a house worth $800000.00 is 3004




# New Model: estimate prices from bedrooms
(bedroom_intercept,bedroom_slope) = simple_linear_regression(train_data['bedrooms'],train_data['price'])
print("Bedroom intercept",bedroom_intercept)
#Bedroom intercept 109473.18046928808
print("Bedroom slope",bedroom_slope)
#Bedroom slope 127588.95217458377


# Test your Linear Regression Algorithm
rss_price_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'],train_data['price'],sqft_intercept,sqft_slope)
print('The RSS of predicting Prices based on Square Feet is : ',rss_prices_on_sqft)
#The RSS of predicting Prices based on Square Feet is :  1201918356321966.8


rss_prices_on_bedrooms = get_residual_sum_of_squares(train_data['bedrooms'],train_data['price'],bedroom_intercept,bedroom_slope)
print('The RSS of predicting Prices based on Bedrooms is : ',rss_prices_on_bedrooms)
#The RSS of predicting Prices based on Bedrooms is :  2143244494226580.5
























