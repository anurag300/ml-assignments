import linearregression as lr
from utility import *
from sklearn.datasets import load_boston
from itertools import combinations



def main():
    dataset = load_boston()
    feature_names = dataset.feature_names
    X = dataset.data
    Y = dataset.target
    X_train,X_test = splitdata(X)
    Y_train,Y_test = splitdata(Y)

########################################################################################################################
    #Histogram Plots
    feature_cols = zip(*X_train)
    for i,col in enumerate(feature_cols):
        plothist(col, feature_names[i])


########################################################################################################################
    #Pearson Corr Coeff
    print '#########Pearson Correlation########### '
    pearson_corr = [pearsonCorr(feature_col,Y_train) for feature_col in zip(*X_train)]
    print 'Pearson Correlation Coefficients:',pearson_corr
    pearson_corr = map(lambda x : abs(x),pearson_corr)


########################################################################################################################
    #Standardizing the data
    std_params = []
    X_train_std = []
    for feature_col in zip(*X_train):
        mu = mean(feature_col)
        sig = sigma(feature_col)
        std_params.append((mu,sig))
        X_train_std.append(standardize(feature_col,mu,sig))

    X_train_std = zip(*X_train_std)
    X_test_std = [standardize(feature_col,std_params[i][0],std_params[i][1]) for i,feature_col in enumerate(zip(*X_test))]
    X_test_std = zip(*X_test_std)

########################################################################################################################
    #Creating linear regression model
    linearreg  = lr.LinearRegression(X_train_std,Y_train)
    W = linearreg.getparam()

    #Predicting target
    Y_predict_train = linearreg.predict(X_train_std)
    Y_predict_test = linearreg.predict(X_test_std)

    #Calculating Mean Squared Loss
    #Training
    print '\n\n#######Linear Regression########'
    Y_predict = Y_predict_train.getT().tolist()[0]
    MSL_train = calculateMSL(Y_train,Y_predict)
    print 'Mean Squared Loss for training data is {}'.format(MSL_train)

    #Testing
    Y_predict = Y_predict_test.getT().tolist()[0]
    MSL_test = calculateMSL(Y_test,Y_predict)
    print 'Mean Squared Loss for testing data is {}'.format(MSL_test)


#######################################################################################################################
    print '\n\n#######Ridge Regression#########'
    #Creating ridge regression model
    L = [0.01,0.1,1.0]
    for l in L:
        ridgeregression = lr.RidgeRegression(X_train_std,Y_train,l)
        W = ridgeregression.getparam()
        Y_predict_train = ridgeregression.predict(X_train_std)
        Y_predict_test = ridgeregression.predict(X_test_std)
        MSL_train = calculateMSL(Y_train,Y_predict_train.getT().tolist()[0])
        MSL_test = calculateMSL(Y_test, Y_predict_test.getT().tolist()[0])
        print 'Mean Squared Loss for lamda {l} on training data is {y}'.format(l = l, y = MSL_train)
        print 'Mean Squared Loss for lamda {l} on testing data is {y}'.format(l = l, y = MSL_test)


#######################################################################################################################
    print '\n\n##########Predicting lambda using 10 fold cross validation###########'
    kf = k_fold(len(X_train),n_folds=10)
    l_dict = {}
    for l in np.arange(0.0001,10,0.1):
        for train, test in kf:
            X_train_std_subset = map(lambda x : X_train_std[x],train)
            Y_train_subset = map(lambda x : Y_train[x],train)
            X_test_std_subset = map(lambda x : X_train_std[x],test)
            Y_test_subset = map(lambda x : Y_train[x],test)
            ridgeregression = lr.RidgeRegression(X_train_std_subset, Y_train_subset, l)
            W = ridgeregression.getparam()
            Y_predict_test = ridgeregression.predict(X_test_std_subset)
            MSL_test = calculateMSL(Y_test_subset, Y_predict_test.getT().tolist()[0])
            if l in l_dict:
                l_dict[l].append(MSL_test)
            else:
                l_dict[l] = []
                l_dict[l].append(MSL_test)

    for key, val in l_dict.items():
        l_dict[key] = sum(l_dict[key])/len(l_dict[key])

    best_l = min(l_dict.items(), key = lambda k : k[1])[0]
    ridgeregression = lr.RidgeRegression(X_train_std, Y_train, best_l)
    W = ridgeregression.getparam()
    Y_predict_test = ridgeregression.predict(X_test_std)
    Y_predict_train = ridgeregression.predict(X_train_std)
    MSL_test = calculateMSL(Y_test, Y_predict_test.getT().tolist()[0])
    MSL_train = calculateMSL(Y_train, Y_predict_train.getT().tolist()[0])
    print 'Best value of lambda with 10 fold CV is {}'.format(best_l)
    print 'Mean Squared Loss on train data for lambda {l} is {y}'.format(l=best_l,y=MSL_train)
    print 'Mean Squared Loss on test data for lambda {l} is {y}'.format(l=best_l,y=MSL_test)

    plotXY(l_dict.keys(),l_dict.values())
    # for l in l_dict.keys():
    #     ridgeregression = lr.RidgeRegression(X_train_std, Y_train, l)
    #     W = ridgeregression.getparam()
    #     Y_predict_test = ridgeregression.predict(X_test_std)
    #     MSL_test = calculateMSL(Y_test, Y_predict_test.getT().tolist()[0])
       # print 'Mean Squared Loss for Lamda {l} Test is {y}'.format(l=l, y=MSL_test)


########################################################################################################################
    #HFeature Selection - Top 4
    print '\n\n#######Selecting top 4 features with highest correlation with target########'
    ind = np.argpartition(pearson_corr, -4)[-4:]
    ind = ind.tolist()
    ind.reverse()
    X_temp = zip(*X_train_std)
    X_k_top_train =  [X_temp[i] for i in ind]
    X_k_top_train = zip(*X_k_top_train)
    X_temp = zip(*X_test_std)
    X_k_top_test = [X_temp[i] for i in ind]
    X_k_top_test = zip(*X_k_top_test)
    linearreg = lr.LinearRegression(X_k_top_train,Y_train)
    W = linearreg.getparam()
    Y_predict_test = linearreg.predict(X_k_top_test)
    Y_predict_train = linearreg.predict(X_k_top_train)
    MSL_test = calculateMSL(Y_test, Y_predict_test.getT().tolist()[0])
    MSL_train = calculateMSL(Y_train, Y_predict_train.getT().tolist()[0])
    print 'Top 4 correlated features are {}'.format([feature_names[i] for i in ind])
    print 'Mean Squared Loss on train data with these 4 features is {y}'.format(y=MSL_train)
    print 'Mean Squared Loss on test data with these 4 features is {y}'.format(y=MSL_test)

########################################################################################################################
    #Feature Selection using residue
    print '\n\n#######Selecting top features with highest residual correlation with features########'
    residue = Y_train
    X_train_temp = X_train_std
    X_test_temp = X_test_std
    temp_feature_name = feature_names
    ind_result = []
    ind_test_result = []
    ind_feature_result = []
    for i in xrange(0,4):
        pearson_corr_temp = [abs(pearsonCorr(feature_col,residue)) for feature_col in zip(*X_train_temp)]
        ind = np.argpartition(pearson_corr_temp, -1)[-1:][0]
        ind_result.append(zip(*X_train_temp)[ind])
        ind_test_result.append(zip(*X_test_temp)[ind])
        ind_feature_result.append(temp_feature_name[ind])
        linearreg = lr.LinearRegression(zip(*ind_result), Y_train)
        W = linearreg.getparam()
        Y_predict = linearreg.predict(zip(*ind_result))
        residue = (np.matrix(zip(Y_train))- Y_predict).tolist()
        residue = map(lambda x : x[0],residue)
        X_train_temp = [zip(*X_train_temp)[i] for i in xrange(len(zip(*X_train_temp))) if i!=ind]
        X_train_temp = zip(*X_train_temp)
        X_test_temp = [zip(*X_test_temp)[i] for i in xrange(len(zip(*X_test_temp))) if i!=ind]
        X_test_temp = zip(*X_test_temp)
        temp_feature_name = [temp_feature_name[i] for i in xrange(len(temp_feature_name)) if i!=ind]


    X_k_top_train = ind_result
    X_k_top_train = zip(*X_k_top_train)
    X_k_top_test =  ind_test_result
    X_k_top_test = zip(*X_k_top_test)
    linearreg = lr.LinearRegression(X_k_top_train, Y_train)
    W = linearreg.getparam()
    Y_predict_test = linearreg.predict(X_k_top_test)
    Y_predict_train = linearreg.predict(X_k_top_train)
    MSL_test = calculateMSL(Y_test, Y_predict_test.getT().tolist()[0])
    MSL_train = calculateMSL(Y_train, Y_predict_train.getT().tolist()[0])
    print 'Top 4 residual correlated features are {}'.format([feature for feature in ind_feature_result])
    print 'Mean Squared Loss for train data on top 4 features is {y}'.format(y=MSL_train)
    print 'Mean Squared Loss for test data on top 4 features is {y}'.format(y=MSL_test)


########################################################################################################################
    #Selecting features with Brute force
    print '\n\n############Selection with bruteforce##############'
    num_features = len(feature_names)
    result = {}
    for c in combinations(xrange(num_features),4):
        X_train_bf = [zip(*X_train_std)[i] for i in c]
        X_train_bf = zip(*X_train_bf)
        X_test_bf = [zip(*X_test_std)[i] for i in c]
        X_test_bf = zip(*X_test_bf)
        linearreg = lr.LinearRegression(X_train_bf, Y_train)
        W = linearreg.getparam()
        Y_predict_test = linearreg.predict(X_test_bf)
        Y_predict_train = linearreg.predict(X_train_bf)
        MSL_test = calculateMSL(Y_test, Y_predict_test.getT().tolist()[0])
        MSL_train = calculateMSL(Y_train, Y_predict_train.getT().tolist()[0])
        result[c] = (MSL_test,MSL_train)
    top_k,vals =  min(result.items(), key = lambda k : k[1][1])
    top_k_features = [feature_names[i] for i in top_k]
    MSL_test = vals[0]
    MSL_train = vals[1]
    print 'Top 4 bruteforce features are {}'.format([feature for feature in top_k_features])
    print 'Mean Squared Loss for train data on top 4 features is {y}'.format(y=MSL_train)
    print 'Mean Squared Loss for test data on top 4 features is {y}'.format(y=MSL_test)

########################################################################################################################
    #Feature expansion
    print '\n\n############Feature expansion##############'
    num_features = len(feature_names)
    comb = [(i,j) for i in xrange(num_features) for j in xrange(num_features) if i<=j]
    X_train_fs = X_train_std
    X_train_fs = zip(*X_train_fs)
    X_test_fs = X_test_std
    X_test_fs = zip(*X_test_fs)
    X_train_expanded = [f for f in X_train_fs]
    X_test_expanded = [f for f in X_test_fs]
    for i,j in comb:
        fi = zip(*X_train_std)[i]
        fj = zip(*X_train_std)[j]
        fij = tuple(np.array(fi)*np.array(fj))
        mu = mean(fij)
        sig = sigma(fij)
        X_train_expanded.append(standardize(fij,mu,sig))
        fi = zip(*X_test_std)[i]
        fj = zip(*X_test_std)[j]
        fij = tuple(np.array(fi) * np.array(fj))
        X_test_expanded.append(standardize(fij,mu,sig))

    X_train_expanded = zip(*X_train_expanded)
    X_test_expanded = zip(*X_test_expanded)
    linearreg = lr.LinearRegression(X_train_expanded, Y_train)
    W = linearreg.getparam()
    Y_predict_test = linearreg.predict(X_test_expanded)
    Y_predict_train = linearreg.predict(X_train_expanded)
    MSL_test = calculateMSL(Y_test, Y_predict_test.getT().tolist()[0])
    MSL_train = calculateMSL(Y_train, Y_predict_train.getT().tolist()[0])
    print 'Mean Squared Loss for training data with expanded (104) features is {y}'.format(y=MSL_train)
    print 'Mean Squared Loss for test data with expanded (104) features is {y}'.format(y=MSL_test)

if __name__ == '__main__':
    main()

