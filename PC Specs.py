# Importing the libraries
import numpy as sNumpy
#import matplotlib.pyplot as sPyplot
import pandas as sPandas

# Initialize variables
X = [];
X_train = [];
X_test = [];
X_remodeled = [];
Y = [];
Y_train = [];
Y_test = [];
Y_predicted = [];
rmse = 0;
averageRMSE = 0;
best_Y_test = [];
best_Y_predicted = [];
bestAverageRMSE = 0;
bestSL = 0;

# USER SETTINGS (THIS IS WHERE THE MAGIC HAPPENS!)
# For unanswered values (predicting sales)
# This should be your (last record with a logged price - 1) in excel
trainingRows = 160; # Put -1 here if you want to disable this (if you want to train everything)
# For loop amounts 100:
    # Records:115    bestSL:0.07    bestAverageRMSE:825
    # Records:133    bestSL:0.04    bestAverageRMSE:716
    # Records:147    bestSL:0.03    bestAverageRMSE:721
    # Records:161    bestSL:0.07    bestAverageRMSE:777
bestSLToUse = 0.07;
# For answered values (testing)
loopAmount = 100;
arrSLOptions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10];
bestAverageRMSE = 999999999999999999;

####################################################################################################################################

# Importing the dataset
dataset = sPandas.read_excel('PC Specs.xlsx');

# This needs to be revisited. At the moment we are ignoring the outliers if we are testing unanswered values
# Otherwise this removes all the NAN tests we want to predict. We need to somehow add the NAN values after the outliers have been removed
# To remove the outliers
if (trainingRows == -1):
    data_filtered = dataset[(dataset["Sold at"]>(sNumpy.mean(dataset["Sold at"])-2*sNumpy.std(dataset["Sold at"]))) & (dataset["Sold at"]<(sNumpy.mean(dataset["Sold at"])+2*sNumpy.std(dataset["Sold at"])))]    
else:
    data_filtered = dataset; # Use this if you want to predict unanswered values (otherwise they will be seen as outliers and be removed)

# The MAIN function that does all the work
def predict(pSLItem, pRandomState, pTrainingRows):
    # Link the global values so that they can be updated
    global X;
    global X_train;
    global X_test;
    global X_remodeled;
    global Y;
    global Y_train;
    global Y_test;
    global Y_predicted;
    global rmse;
    global averageRMSE;
    global best_Y_test;
    global best_Y_predicted;
    global bestAverageRMSE;
    global bestSL;
    
    # Setting data_filtered values to use
        # DATA COLUMNS:
            # 0:  Sell Price
            # 1:  Brand
            # 2:  Model Simplified
            # 3:  Model
            # 4:  Processor
            # 5:  Generation Simplified
            # 6:  Generation
            # 7:  RAM
            # 8:  RAM Type
            # 9:  Graphics Card Simplified
            # 0:  Graphics Card Name
            # 11: SSD
            # 12: Hard Drive Size
            # 13: Date
            # 14: Date Simplified
            # 15: Screen Size
            # 16: Seller
            # 17: Link
    # All usable fields
    X = data_filtered.iloc[:, [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15]].values
    Y = data_filtered.iloc[:, 0].values
    
    # Transform categorical data into numbers
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    
    # Copy this line for every categorical column in (select by the index above)
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 10] = labelencoder_X.fit_transform(X[:, 10])
    
    # Transform non-progressive categorical numbers into multiple columns
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder(categorical_features = [0, 1, 10])
    X = onehotencoder.fit_transform(X).toarray()
    
    # Automatic backwards elimination
    import statsmodels.api as sm
    def backwardElimination(x, sl):
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        x = sNumpy.delete(x, j, 1)
        regressor_OLS.summary()
        return x

    SL = pSLItem
    X_remodeled = backwardElimination(X[:, :], SL);
    
    # Splitting the data_filtered into training and testing datasets
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X_remodeled, Y, test_size = 0.2, random_state = pRandomState)
    
    # EIE DATASET SPLITTING
    if (pTrainingRows != -1):
        X_train = X_remodeled[0:pTrainingRows, :];
        Y_train = Y[0:pTrainingRows];
        X_test = X_remodeled[pTrainingRows:, :]; # Comment these out if you want to test unanswered records
        #Y_test = Y[99:]; # Comment these out if you want to test unanswered records
    
    # Fitting multiple linear regression to the training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    
    # Predicting the test set results
    Y_predicted = regressor.predict(X_test);
    
    # Accuracy check
    if (trainingRows == -1):
        rmse = sNumpy.sqrt(((Y_predicted - Y_test) ** 2).mean());
        averageRMSE = averageRMSE + rmse;   
    
# Main usage for predicting unanswered values
if (trainingRows > -1):
    predict(bestSLToUse, 1, trainingRows); # You never need to change the randomState=1 here, it won't be used anyways

# For looping through many times and picking the best SL value
if (trainingRows == -1):
    bestSL = 0;

    for SLItem in arrSLOptions:
        averageRMSE = 0;
        for index in range(loopAmount):
            predict(SLItem, index, -1);
        
        averageRMSE = averageRMSE / loopAmount;
        if averageRMSE < bestAverageRMSE:
            bestAverageRMSE = averageRMSE;
            bestSL = SLItem;
            best_Y_test = Y_test;
            best_Y_predicted = Y_predicted;

# To fix the view variable error
X = sPandas.DataFrame(X)
Y = sPandas.DataFrame(Y)

# Delete variables we dont need to see from the variable explorer
if (trainingRows == -1):
    del SLItem, arrSLOptions, rmse, averageRMSE, index, loopAmount, trainingRows, bestSLToUse, X_test, X_train, Y_test, Y_train, Y_predicted
else:
    del rmse, averageRMSE, trainingRows, bestSLToUse, bestAverageRMSE, bestSL, best_Y_test, best_Y_predicted, Y_test, loopAmount, arrSLOptions