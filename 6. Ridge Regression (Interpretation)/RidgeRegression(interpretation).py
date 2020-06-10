import turicreate 

def polynomial_sframe(features,degree):
	poly_sframe = turicreate.SFrame()
	poly_sframe['power_1'] = features
	if degree > 1:
		for power in range(2,degree+1):
			name = 'power_'+str(degree)
			poly_sframe[name] = ply_sframe['power_1'].apply(lambda x:x**power)
	return poly_sframe
 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
sales = turicreate.SFrame('home_data.sframe')
sales = sales.sort(['sqft_living','price'])
sales.head()

#-----------------------------------------------------------------------------------------------------

l2_small_penalty = 1e-5
poly_features = polynomial_sframe(sales['sqft_living'],1)
 
#Model without l2 penalty
simple_model = turicreate.linear_regression.create(poly_features,target='price',features=['power_1'],validation_set=None,verbose=True)
simple_model.coefficients
#+-------------+-------+--------------------+-------------------+
#|     name    | index |       value        |       stderr      |
#+-------------+-------+--------------------+-------------------+
#| (intercept) |  None | -43579.08525145019 | 4402.689697427721 |
#|   power_1   |  None | 280.6227708858474  | 1.936398555132125 |
#+-------------+-------+--------------------+-------------------+
#[2 rows x 4 columns]


#Model with l2 penalty
penalty_model = turicreate.linear_regression.create(poly_features,target='price',l2_penalty=l2_small_penalty,features=['power_1'],validation_set=None,verbose=True)
penalty_model.coefficients
#+-------------+-------+---------------------+-------------------+
#|     name    | index |        value        |       stderr      |
#+-------------+-------+---------------------+-------------------+
#| (intercept) |  None | -43580.738672005245 | 4402.689697410518 |
#|   power_1   |  None |  280.62356583789546 | 1.936398555124559 |
#+-------------+-------+---------------------+-------------------+
#[2 rows x 4 columns]


#-----------------------------------------------------------------------------------------------------
#Observe Overfitting

#Set-1
set_1_poly = polynomial_sframe(set_1['sqft_living'],15)
poly_features = set_1_poly.column_names()
set_1_poly['price'] = set_1['price']
set_1_model = turicreate.linear_regression.create(set_1_poly,target='price',l2_penalty=l2_small_penalty,features=poly_features,validation_set=None,verbose=True)
set_1_model.coefficients.print_rows(16)
3+-------------+-------+-------------------------+------------------------+
#|     name    | index |          value          |         stderr         |
#+-------------+-------+-------------------------+------------------------+
#| (intercept) |  None |   -287005.73519421555   |    5641895.2493569     |
#|   power_1   |  None |     1739.96829615449    |   34093.31466638047    |
#|   power_2   |  None |   -2.0313513771808602   |   85.56290578215192    |
#|   power_3   |  None |  0.0011967016020887613  |  0.11837016391522585   |
#|   power_4   |  None | -3.1225474621588606e-07 | 0.00010069803032260626 |
#|   power_5   |  None |  1.4832830285972895e-11 | 5.5327072824075065e-08 |
#|   power_6   |  None |  6.725717335693129e-15  | 1.9875776076190858e-11 |
#|   power_7   |  None | -2.3056530344189954e-19 | 4.533549927017628e-15  |
#|   power_8   |  None | -1.4241409217545815e-22 | 5.7918839505940755e-19 |
#|   power_9   |  None |  -8.636422912959192e-27 | 2.0187808383469935e-23 |
#|   power_10  |  None |  1.6190492240520426e-30 | 4.975009694800578e-27  |
#|   power_11  |  None |  3.5815422415974634e-34 |          nan           |
#|   power_12  |  None |  1.6559015534441415e-38 | 8.950349693169149e-35  |
#|   power_13  |  None | -4.9010062341855645e-42 | 1.8395013585944686e-38 |
#|   power_14  |  None |  -9.217329552330147e-46 | 1.2642603452907349e-42 |
#|   power_15  |  None |   9.56563556259405e-50  | 3.2267726878482077e-47 |
#+-------------+-------+-------------------------+------------------------+
#[16 rows x 4 columns]
plt.plot(set_1_poly['power_1'],set_1_poly['price'],'.',set_1_poly['power_1'],set_1_model.predict(set_1_poly),'-')
plt.show()


#Set-2
set_2_poly = polynomial_sframe(set_2['sqft_living'],15)
poly_features = set_2_poly.column_names()
set_2_poly['price'] = set_2['price']
set_2_model = turicreate.linear_regression.create(set_2_poly,target='price',l2_penalty=l2_small_penalty,features=poly_features,validation_set=None,verbose=True)
set_2_model.coefficients.print_rows(16)
#+-------------+-------+-------------------------+-----------------------+
#|     name    | index |          value          |         stderr        |
#+-------------+-------+-------------------------+-----------------------+
#| (intercept) |  None |    215405.33193181752   |   785470.8285640263   |
#|   power_1   |  None |    39.98929675316289    |    3422.51988065489   |
#|   power_2   |  None |    0.0864523992336397   |   6.047880635777882   |
#|   power_3   |  None |  -2.712865476815411e-05 |  0.005552839691472001 |
#|   power_4   |  None |   1.65625055794393e-11  | 2.651864874370765e-06 |
#|   power_5   |  None |  2.5672323181415974e-12 |          nan          |
#|   power_6   |  None | -3.6078002968536146e-16 |          nan          |
#|   power_7   |  None | -1.6175857917445737e-20 |          nan          |
#|   power_8   |  None |  2.2289781517099744e-24 |          nan          |
#|   power_9   |  None |  2.342423264001976e-28  |          nan          |
#|   power_10  |  None |  3.063194053640923e-33  |          nan          |
#|   power_11  |  None | -1.1369519552326114e-36 |          nan          |
#|   power_12  |  None | -1.1542261820822679e-40 | 5.692394346119278e-37 |
#|   power_13  |  None |  -3.71056468964347e-45  |          nan          |
#|   power_14  |  None |  3.1278078165604048e-49 |          nan          |
#|   power_15  |  None |  3.700048826460635e-53  |          nan          |
#+-------------+-------+-------------------------+-----------------------+
#[16 rows x 4 columns]
plt.plot(set_2_poly['power_1'],set_2_poly['price'],'.',set_2_poly['power_1'],set_2_model.predict(set_2_poly),'-')
plt.show()


#Set-3
set_3_poly = polynomial_sframe(set_3['sqft_living'],15)
poly_features = set_3_poly.column_names()
set_3_poly['price'] = set_3['price']
set_3_model = turicreate.linear_regression.create(set_3_poly,target='price',l2_penalty=l2_small_penalty,features=poly_features,validation_set=None,verbose=True)
set_3_model.coefficients.print_rows(16)
#+-------------+-------+-------------------------+------------------------+
#|     name    | index |          value          |         stderr         |
#+-------------+-------+-------------------------+------------------------+
#| (intercept) |  None |    398022.4748564027    |   931917.0586354265    |
#|   power_1   |  None |    -532.6847186475985   |   4352.282339178828    |
#|   power_2   |  None |    0.7741325066273258   |   8.361620850971372    |
#|   power_3   |  None |  -0.0003868608915448352 |  0.00865177633456504   |
#|   power_4   |  None |  6.616483407403692e-08  | 5.261958953222106e-06  |
#|   power_5   |  None |  8.526586051149284e-12  | 1.8560584831461068e-09 |
#|   power_6   |  None |  -3.027777313142036e-15 | 2.2619375402220939e-13 |
#|   power_7   |  None | -4.6697858681820307e-20 |          nan           |
#|   power_8   |  None |  3.818766947996313e-23  |          nan           |
#|   power_9   |  None |  2.7856663833595172e-27 | 2.110372696432382e-24  |
#|   power_10  |  None | -2.2134489799612836e-31 | 4.114085270043001e-28  |
#|   power_11  |  None |  -5.181072875327423e-35 |  8.29850975764616e-33  |
#|   power_12  |  None | -2.4314099875838077e-39 |          nan           |
#|   power_13  |  None |  3.980764467503751e-43  | 3.7699265748248206e-40 |
#|   power_14  |  None |  6.623144209338534e-47  |  2.51287484615773e-44  |
#|   power_15  |  None | -5.0798454033857593e-51 |  5.92099016197581e-49  |
#+-------------+-------+-------------------------+------------------------+
#[16 rows x 4 columns]
plt.plot(set_3_poly['power_1'],set_3_poly['price'],'.',set_3_poly['power_1'],set_3_model.predict(set_3_poly),'-')
plt.show()



#Set-4
set_4_poly = polynomial_sframe(set_4['sqft_living'],15)
poly_features = set_4_poly.column_names()
set_4_poly['price'] = set_4['price']
set_4_model = turicreate.linear_regression.create(set_4_poly,target='price',l2_penalty=l2_small_penalty,features=poly_features,validation_set=None,verbose=True)
set_4_model.coefficients.print_rows(16)
#+-------------+-------+-------------------------+------------------------+
#|     name    | index |          value          |         stderr         |
#+-------------+-------+-------------------------+------------------------+
#| (intercept) |  None |     8023.15556692332    |   602923.2226941311    |
#|   power_1   |  None |    594.5233926808373    |   2594.208268600711    |
#|   power_2   |  None |   -0.45289003742333006  |   4.603804208760092    |
#|   power_3   |  None |  0.0002059342138077537  |  0.004433354898394911  |
#|   power_4   |  None |  -4.118615235462598e-08 | 2.5657345771266448e-06 |
#|   power_5   |  None |  3.1209121174153316e-12 | 9.280271218479627e-10  |
#|   power_6   |  None |  1.1197498191725057e-16 | 2.0770773182350943e-13 |
#|   power_7   |  None | -1.4012118222676318e-20 | 2.5955309146276807e-17 |
#|   power_8   |  None | -1.3265010240938225e-24 | 1.0086475847076797e-21 |
#|   power_9   |  None |  -2.745248121059669e-29 | 2.267779007745667e-25  |
#|   power_10  |  None |  4.527757144161193e-33  | 2.8426674076433565e-29 |
#|   power_11  |  None |  6.122631582005839e-37  | 1.6407575552314944e-33 |
#|   power_12  |  None |  4.251740293563371e-41  | 2.032997768463234e-37  |
#|   power_13  |  None |  1.4039561012657203e-45 | 1.2531967412780138e-41 |
#|   power_14  |  None |  -1.002468210178704e-49 | 9.188537383531319e-46  |
#|   power_15  |  None |  -2.509662372537554e-53 | 2.921669716705183e-50  |
#+-------------+-------+-------------------------+------------------------+
#[16 rows x 4 columns]
plt.plot(set_4_poly['power_1'],set_4_poly['price'],'.',set_4_poly['power_1'],set_4_model.predict(set_4_poly),'-')
plt.show()

#-----------------------------------------------------------------------------------------------------
#Ridge Regressin comes to resue
l2_penalty = 1e5
#Set-1
set_1_poly = polynomial_sframe(set_1['sqft_living'],15)
poly_features = set_1_poly.column_names()
set_1_poly['price'] = set_1['price']
set_1_model = turicreate.linear_regression.create(set_1_poly,target='price',l2_penalty=l2_penalty,features=poly_features,validation_set=None,verbose=True)
set_1_model.coefficients.print_rows(16)
#+-------------+-------+------------------------+------------------------+
#|     name    | index |         value          |         stderr         |
#+-------------+-------+------------------------+------------------------+
#| (intercept) |  None |   522426.50036967633   |   7752563.465713456    |
#|   power_1   |  None |   2.1589232072733404   |   46847.83641415211    |
#|   power_2   |  None |  0.001182030545879377  |   117.57252271966665   |
#|   power_3   |  None | 3.036535691044597e-07  |  0.16265318082683217   |
#|   power_4   |  None | 4.923296783561138e-11  | 0.00013836979178890793 |
#|   power_5   |  None | 6.324118771327531e-15  | 7.602527599031266e-08  |
#|   power_6   |  None | 7.442139842430203e-19  | 2.7311427924604375e-11 |
#|   power_7   |  None | 8.584722543156909e-23  | 6.2295792425765095e-15 |
#|   power_8   |  None | 9.956285801877108e-27  | 7.958663875963661e-19  |
#|   power_9   |  None | 1.1692055792436226e-30 | 2.7740193465015607e-23 |
#|   power_10  |  None | 1.3905583349548148e-34 | 6.836191864050926e-27  |
#|   power_11  |  None | 1.6716538371622826e-38 |          nan           |
#|   power_12  |  None | 2.026661088418738e-42  | 1.229873136062427e-34  |
#|   power_13  |  None | 2.4730798090826842e-46 | 2.5276702947285827e-38 |
#|   power_14  |  None | 3.0327876637694446e-50 | 1.7372280290330524e-42 |
#|   power_15  |  None |  3.7332001686327e-54   | 4.4339284843733115e-47 |
#+-------------+-------+------------------------+------------------------+
#[16 rows x 4 columns]
plt.plot(set_1_poly['power_1'],set_1_poly['price'],'.',set_1_poly['power_1'],set_1_model.predict(set_1_poly),'-')
plt.show()


#Set-2
set_2_poly = polynomial_sframe(set_2['sqft_living'],15)
poly_features = set_2_poly.column_names()
set_2_poly['price'] = set_2['price']
set_2_model = turicreate.linear_regression.create(set_2_poly,target='price',l2_penalty=l2_penalty,features=poly_features,validation_set=None,verbose=True)
set_2_model.coefficients.print_rows(16)
#+-------------+-------+------------------------+------------------------+
#|     name    | index |         value          |         stderr         |
#+-------------+-------+------------------------+------------------------+
#| (intercept) |  None |    519224.051697027    |   1109752.1468968608   |
#|   power_1   |  None |   2.269364539056681    |   4835.505848508227    |
#|   power_2   |  None |  0.001249076566535651  |   8.544745744409738    |
#|   power_3   |  None | 2.430743881380568e-07  |  0.007845327343665758  |
#|   power_4   |  None | 2.167718221403383e-11  | 3.7466862302110864e-06 |
#|   power_5   |  None | 1.4892185233101673e-15 |          nan           |
#|   power_6   |  None | 1.018459240791867e-19  |          nan           |
#|   power_7   |  None | 7.298402094794519e-24  |          nan           |
#|   power_8   |  None | 5.457774125637055e-28  |          nan           |
#|   power_9   |  None | 4.209363505170298e-32  |          nan           |
#|   power_10  |  None | 3.3166686614312956e-36 |          nan           |
#|   power_11  |  None | 2.6522501876214835e-40 |          nan           |
#|   power_12  |  None | 2.143013842188718e-44  | 8.042497081830823e-37  |
#|   power_13  |  None | 1.744298046765339e-48  |          nan           |
#|   power_14  |  None | 1.4272300833712201e-52 |          nan           |
#|   power_15  |  None | 1.1722179843187817e-56 |          nan           |
#+-------------+-------+------------------------+------------------------+
#[16 rows x 4 columns]
plt.plot(set_2_poly['power_1'],set_2_poly['price'],'.',set_2_poly['power_1'],set_2_model.predict(set_2_poly),'-')
plt.show()


#Set-3
set_3_poly = polynomial_sframe(set_3['sqft_living'],15)
poly_features = set_3_poly.column_names()
set_3_poly['price'] = set_3['price']
set_3_model = turicreate.linear_regression.create(set_3_poly,target='price',l2_penalty=l2_penalty,features=poly_features,validation_set=None,verbose=True)
set_3_model.coefficients.print_rows(16)
#+-------------+-------+------------------------+------------------------+
#|     name    | index |         value          |         stderr         |
#+-------------+-------+------------------------+------------------------+
#| (intercept) |  None |   521074.74657414114   |   1298313.9822561801   |
#|   power_1   |  None |   2.1544876059957847   |   6063.446272736466    |
#|   power_2   |  None |  0.001215165497846629  |   11.649115298991337   |
#|   power_3   |  None | 2.837628554151892e-07  |  0.012053349686469522  |
#|   power_4   |  None | 3.473583313584255e-11  | 7.330775651890886e-06  |
#|   power_5   |  None | 3.1722305926958025e-15 | 2.585795225255663e-09  |
#|   power_6   |  None | 2.7431772943085163e-19 | 3.1512516143449613e-13 |
#|   power_7   |  None | 2.4322156404295995e-23 |          nan           |
#|   power_8   |  None | 2.237483337011162e-27  |          nan           |
#|   power_9   |  None | 2.121832395147732e-31  | 2.9400968188755096e-24 |
#|   power_10  |  None | 2.0565498678294288e-35 | 5.7315985159797475e-28 |
#|   power_11  |  None | 2.0235219198213124e-39 | 1.1561191149368433e-32 |
#|   power_12  |  None | 2.0117044091049227e-43 |          nan           |
#|   power_13  |  None | 2.0142655740396315e-47 |  5.25212875847678e-40  |
#|   power_14  |  None | 2.026847422085041e-51  | 3.5008486197297373e-44 |
#|   power_15  |  None | 2.0465727997001755e-55 | 8.248914691346814e-49  |
#+-------------+-------+------------------------+------------------------+
#[16 rows x 4 columns]
plt.plot(set_3_poly['power_1'],set_3_poly['price'],'.',set_3_poly['power_1'],set_3_model.predict(set_3_poly),'-')
plt.show()



#Set-4
set_4_poly = polynomial_sframe(set_4['sqft_living'],15)
poly_features = set_4_poly.column_names()
set_4_poly['price'] = set_4['price']
set_4_model = turicreate.linear_regression.create(set_4_poly,target='price',l2_penalty=l2_penalty,features=poly_features,validation_set=None,verbose=True)
set_4_model.coefficients.print_rows(16)
#+-------------+-------+------------------------+------------------------+
#|     name    | index |         value          |         stderr         |
#+-------------+-------+------------------------+------------------------+
#| (intercept) |  None |   525648.9750573988    |   870942.2902681681    |
#|   power_1   |  None |   2.272752287682051    |   3747.418586385983    |
#|   power_2   |  None | 0.0011668798432637856  |   6.650345567395535    |
#|   power_3   |  None | 1.6760767303301558e-07 |  0.006404125970676848  |
#|   power_4   |  None |  8.79708204951424e-12  | 3.7062874089302544e-06 |
#|   power_5   |  None | 3.6960329799271555e-16 | 1.3405654924379619e-09 |
#|   power_6   |  None | 1.6552620476617504e-20 | 3.000406035986225e-13  |
#|   power_7   |  None | 8.176069333134225e-25  | 3.7493291917774885e-17 |
#|   power_8   |  None |  4.37956567807251e-29  | 1.4570243845863953e-21 |
#|   power_9   |  None | 2.4928799743730573e-33 | 3.2758808559445305e-25 |
#|   power_10  |  None | 1.4860835921350134e-37 | 4.106325928897879e-29  |
#|   power_11  |  None | 9.193162777228921e-42  | 2.3701278855086772e-33 |
#|   power_12  |  None | 5.8665415231768825e-46 | 2.936731686438455e-37  |
#|   power_13  |  None | 3.845051092291467e-50  | 1.8102836296935734e-41 |
#|   power_14  |  None | 2.5788270052572663e-54 | 1.327314240321987e-45  |
#|   power_15  |  None | 1.763850336639221e-58  | 4.2204473450266574e-50 |
#+-------------+-------+------------------------+------------------------+
#[16 rows x 4 columns]
plt.plot(set_4_poly['power_1'],set_4_poly['price'],'.',set_4_poly['power_1'],set_4_model.predict(set_4_poly),'-')
plt.show()

#------------------------------------------------------------------------------------------------------
# Selecting an L2 penalty via cross-validation

import turicreate_cross_validation.cross_validation as tcv
(train_valid, test) = salse.random_split(0.9,seed=1)
train_valid_shuffled = tcv.shuffle_sframe(train_valid,random_seed=1)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in range(k):
	start=(n*i)/k
	end=(n*(i+1))/k-1
	print(i,(start,end))

#0 (0.0, 1938)
#1 (1939, 3878)
#2 (3879, 5817)
#3 (5818, 7757)
#4 (7758, 9697.0)
#5 (9698.0, 11636)
#6 (11637, 13576)
#7 (13577, 15515)
#8 (15516, 17455)
#9 (17456, 19395)
 


train_valid_shuffled[0:9]
n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print(first_two.append(last_two))


validation4 = train_valid_shuffled[5818:7758]				#3 (5818, 7757)

print(int(round(validation4['price'].mean(),0)))			#546205

train4 = train_valid_shuffled[0:5818]
train_41 = train_valid[7758:19397]
train4 = train4.append(train_41)

import numpy as np

def k_fold_cross_validation(k,l2_penalty,data,output_name,features_list):
	rss_sum = 0
	n = len(data)
	for i in range(k):
		start = int(round((n*i)/k))
		end = int(round((n*(i+1))/k-1))
		validation_set = data[start:end+1]
		training_set = data[0:start].append(data[end+1:n])
		model = turicreate.linear_regression.create(training_set,target=output_name,features=features_list,l2_penalty=l2_penalty,validation_set=None,verbose=False)
		predictions = model.predict(validation_set)
		residuals = validation_set['price']-predictions
		rss = sum(residuals*residuals)
		rss_sum += rss
	validation_error = rss_sum/k
	return validation_error

poly_data = polynomial_sframe(train_valid_shuffled['sqft_living'],15)
my_features = poly_data.column_names()
poly_data['price'] = train_valid_shuffled['price']

val_err_dict = {}
for l2_penalty in np.logspace(1,7,num=13):
	val_err = k_fold_cross_validation(10,l2_penalty,poly_data,'price',my_features)
	print(l2_penalty)
	val_err_dict[l2_penalty] = val_err
print(val_err_dict)
#{10.0: 635589143706849.8, 31.622776601683793: 337831527234540.94, 100.0: 170836445261608.22, 316.22776601683796: 121505814225582.55, 1000.0: 121047133815812.92, 3162.2776601683795: 123240200267920.6, 10000.0: 135066542330447.28, 31622.776601683792: 170720643116852.78, 100000.0: 227518668807885.8, 316227.7660168379: 250295246490498.3, 1000000.0: 255722415402776.5, 3162277.6601683795: 259724594772267.16, 10000000.0: 261741601056498.66}
print(min(val_err_dict.items(),key=lambda x:x[1]))
#(1000.0, 121047133815812.92)
	
l2_penalty = turicreate.SArray(val_err_dict.keys())
validation_error = turicreate.SArray(val_err_dict.values())
sf = turicreate.SFrame({'l2_penalty':l2_penalty,'validation_error':validation_error})
print(sf)

plt.plot(sf['l2_penalty'],sf['validation_error'],'k.')
plt.xscale('log')
plt.show()

poly_data = polynomial_sframe(train_valid_shuffled['sqft_living'],15)
features_list = poly_data.column_names()
poly_data['price'] = train_valid_shuffled['price']
l2_penalty_best = 1000

model = turicreate.linear_regression.create(poly_data,target='price',features=features_list,l2_penalty=l2_penalty_best,validation_set=None)


poly_test = polynomial_sframe(test['sqft_living'], 15)
predictions = model.predict(poly_test)
errors = predictions-test['price']
rss = (errors*errors).sum()
print(rss)
#138704957784694.61
































