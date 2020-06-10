import turicreate 

#Importing Data
sales = turicreate.SFrame('https://courses.cs.washington.edu/courses/cse416/18sp/notebooks/Philadelphia_Crime_Rate_noNA.csv')

# Exploring the data 
#The house price in a town is correlated with the crime rate of that town. Low crime towns tend to be associated with higher house prices and vice versa.
turicreate.show(sales['CrimeRate'],sales['HousePrice'])


# Fit the regression model using crime as the feature
crime_model = turicreate.linear_regression.create(sales,target='HousePrice',features=['CrimeRate'],validation_set=None,verbose=False)

#Let's see what our fit looks like 
import matplotlib.pylplot as plt
plt.plot(sales['CrimeRate'],sales['HousePrice'],'.',
		 sales['CrimeRate'],crime_model.predict(sales),'-')
plt.show()


# Remove Center City and redo the analysis
#Center City is the one observation with an extremely high crime rate, yet house prices are not very low. This point does not follow the trend of the rest of the data very well. A question is how much including Center City is influencing our fit on the other datapoints. Let's remove this datapoint and see what happens.
sales_noCC = sales[sales['MilesPhila']!=0.0]
turicreate.show(sales_noCC['CrimeRate'],sales_noCC['HousePrice'])

#Fitting a regression model for sales_noCC
crime_model_noCC = turicreatelinear_regression.create(sales_noCC,target='HosePrice',features=['CrimeRate'],validation_set=None,verbose=False)

#Lets visualize the fit
plt.plot(sales_noCC['CrimeRate'],sales_noCC['HousePrice'],'.',
		 sales_noCC['CrimeRate'],crime_model_noCC.predict(sales_noCC),'-')

# Compare coefficients for full-data fit versus no-Center-City fit
#Visually, the fit seems different, but let's quantify this by examining the estimated coefficients of our original fit and that of the modified dataset with Center City removed.
crime_model.coefficients
crime_model_noCC.coefficients

#Above: We see that for the "no Center City" version, per unit increase in crime, the predicted decrease in house prices is 2,287.  In contrast, for the original dataset, the drop is only 576 per unit increase in crime.  This is significantly different!

#High leverage points:Center City is said to be a "high leverage" point because it is at an extreme x value where there are not other observations. As a result, recalling the closed-form solution for simple regression, this point has the potential to dramatically change the least squares line since the center of x mass is heavily influenced by this one point and the least squares line will try to fit close to that outlying (in x) point. If a high leverage point follows the trend of the other data, this might not have much effect. On the other hand, if this point somehow differs, it can be strongly influential in the resulting fit.

#Influential observations:An influential observation is one where the removal of the point significantly changes the fit. As discussed above, high leverage points are good candidates for being influential observations, but need not be. Other observations that are not leverage points can also be influential observations (e.g., strongly outlying in y even if x is a typical value).

# Plotting the two models
#Confirm the above calculations by looking at the plots. The orange line is the model trained removing Center City, and the green line is the model trained on all the data. Notice how much steeper the green line is, since the drop in value is much higher according to this model.

plt.plot(sales_noCC['CrimeRate'], sales_noCC['HousePrice'], '.',
         sales_noCC['CrimeRate'], crime_model.predict(sales_noCC), '-',
         sales_noCC['CrimeRate'], crime_model_noCC.predict(sales_noCC), '-')

# Remove high-value outlier neighborhoods and redo analysis
sales_nohighend = sales_noCC[sales_noCC['HousePrice'] < 350000]
crime_model_nohighend = turicreate.linear_regression.create(
    sales_nohighend,
    target='HousePrice',
    features=['CrimeRate'],
    validation_set=None,
    verbose=False
)

### Do the coefficients change much?
crime_model_noCC.coefficients
crime_model_nohighend.coefficients

### Compare the two models
#Confirm the above calculations by looking at the plots. The orange line is the no high-end model, and the green line is the no-city-center model.
plt.plot(sales_nohighend['CrimeRate'], sales_nohighend['HousePrice'], '.',
         sales_nohighend['CrimeRate'], crime_model_nohighend.predict(sales_nohighend), '-',
         sales_nohighend['CrimeRate'], crime_model_noCC.predict(sales_nohighend), '-')         
