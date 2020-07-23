# linear-regression-grades
Predicts students' final grades using linear regression model


My comments 

# X is used for training and testing. X contains all features which machine studies to predict an output y. X stores every feature except G3 because we want to predict G3
X = np.array(data.drop([forecast], 1))
# y here is used to store output G3 to train a machine
y = np.array(data[forecast])
# We divide X and y into 4 variables because when training a machine it should also predict for new students.
# When we dont split , machine simply memorizes but it fails to predict a G3 for new student. (test_size=0.1) means that 10 % of all data is used for testing
# train_test_split is for splitting data arrays into two subsets: for training data and for testing data
X_train, y_train, X_test, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()
# here sklearn plots all x and y training datas and find the best coefficients to minimize a cost function
linear.fit(X_train, y_train)
# after plotting the most appropriate hypotheses, score() uses to say how much accurate it is for predicting
acc = linear.score(X_test, y_test)
# the accuracy is 0.714 (71%)
print(acc)
# To find out what are the coefficients and intecept is of hypotheses:
print(f'{linear.coef_} are coefficients' + f' and {linear.intercept_} is intercept of our hypotheses')

# To find out how well does hypotheses predict G3 on new data sets we use test. Below method predict() predicts what are G3 values using X_test features and my hypothese. 
# Note that it did not memorize any data from X_test
predictions = linear.predict(X_test)

# Now predictions[x] is machine's prediction based on X_test features(attributes) and hypotheses({linear}). y_test[x] are actual values of G3 
for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])

Terminal : 11.52529882117441 [10 12  1  1  2  4  5] 12
11.52529882117441 is prediction
[10 12  1  1  2  4  5] are attribute of X_test
12 is actual value of G3 (y_test)
