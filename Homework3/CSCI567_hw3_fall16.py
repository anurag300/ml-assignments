import copy
import time
import scipy.io
import linearreg as lr
from libsvm import run_svm
from svmutil import *
from utility import *


def fx(x, mu, sigma):
    return 2 * (x ** 2) + generate_normal_data(mu, sigma)


def gx(i, x):
    if i == 1:
        return []
    elif i == 2:
        return [0 * x, 0 * x ** 2, 0 * x ** 3, 0 * x ** 4]
    elif i == 3:
        return [1 * x, 0 * x ** 2, 0 * x ** 3, 0 * x ** 4]
    elif i == 4:
        return [1 * x, 1 * x ** 2, 0 * x ** 3, 0 * x ** 4]
    elif i == 5:
        return [1 * x, 1 * x ** 2, 1 * x ** 3, 0 * x ** 4]
    elif i == 6:
        return [1 * x, 1 * x ** 2, 1 * x ** 3, 1 * x ** 4]


def standardize_data(X):
    X_train_std = [standardize(feature_col, mean(feature_col), sigma(feature_col)) for feature_col in zip(*X)]
    return zip(*X_train_std)
    return X


def gen_dataset(S, D):
    datasets = []
    for d in range(D):
        sample = []
        for s in range(S):
            x = generate_uniform_data(-1, 1)
            sample.append((x, fx(x, mu=0, sigma=math.sqrt(0.1))))
        datasets.append(sample)
    return datasets


def train_lr(datasets):
    model = {}
    sse = {}
    mse = {}
    y_pred_dict = {}
    for i in range(1, 7):
        model[i] = {}
        sse[i] = {}
        mse[i] = {}
        y_pred_dict[i] = {}
        for j, dataset in enumerate(datasets):
            if i == 1:
                y_train = zip(*dataset)[1]
                y_predict = [1] * len(y_train)
                sse_train = calculate_sse(y_train, y_predict)
                model[i][j] = None
                sse[i][j] = sse_train
                mse[i][j] = sse_train / len(y_predict)
                y_pred_dict[i][j] = y_predict


            else:
                x_train = map(lambda x: gx(i, x), zip(*dataset)[0])
                y_train = zip(*dataset)[1]
                x_train_std = standardize_data(x_train)
                linearreg = lr.LinearRegression(x_train_std, y_train)
                w = linearreg.getparam()
                y_predict_train = linearreg.predict(x_train_std)
                y_predict = y_predict_train.getT().tolist()[0]
                sse_train = calculate_sse(y_train, y_predict)
                model[i][j] = copy.deepcopy(linearreg)
                sse[i][j] = sse_train
                mse[i][j] = sse_train / len(y_predict)
                y_pred_dict[i][j] = y_predict

    return {'model': model,
            'sse': sse,
            'mse': mse,
            'y_predict_dict': y_pred_dict,
            }


def train_rr(datasets, L):
    y_predict_dict = {}
    model = {}
    for l in L:
        y_predict_dict[l] = {}
        model[l] = {}
        for i, dataset in enumerate(datasets):
            x_train = map(lambda x: gx(4, x), zip(*dataset)[0])
            y_train = zip(*dataset)[1]
            x_train_std = standardize_data(x_train)
            ridgereg = lr.RidgeRegression(x_train_std, y_train, l)
            w = ridgereg.getparam()
            y_predict_train = ridgereg.predict(x_train_std)
            y_predict_dict[l][i] = y_predict_train.getT().tolist()[0]
            model[l][i] = copy.deepcopy(ridgereg)

    return {'y_predict_dict': y_predict_dict,
            'model': model}


def calculate_biassq_lr(y_pred_dict, datasets, S, D):
    biassq = []
    for g in range(1, 7):
        biassq_temp = []
        for i, dataset in enumerate(datasets):
            y_pred = y_pred_dict[g][i]
            y_true = zip(*dataset)[1]
            biassq_temp.append(mean(np.power(np.subtract(np.array(y_pred), np.array(y_true)), 2)))

        biassq.append(mean(biassq_temp))
    return biassq


def calculate_variance_lr(model_dict, D, y_true, x_true):
    variance = []
    for g in range(1, 7):
        for s in x_true:
            y_pred = []
            var_temp = []
            for d in range(D):
                model = model_dict[g][d]
                if model == None:
                    y_pred.append([1])
                else:
                    y_pred.append(model.predict([gx(g, s)]).getT().tolist()[0])
            y_pred_exp = mean(y_pred)
            var_temp.append(
                sum(np.power(np.subtract(np.array(y_pred), np.array(np.repeat(y_pred_exp, D))), 2)) / (D - 1))
        variance.append(mean(var_temp))
    return variance


def calculate_biassq_rr(y_pred_dict, datasets, S, D, L):
    biassq = []
    for l in L:
        biassq_temp = []
        for i, dataset in enumerate(datasets):
            y_pred = y_pred_dict[l][i]
            y_true = zip(*dataset)[1]
            biassq_temp.append(mean(np.power(np.subtract(np.array(y_pred), np.array(y_true)), 2)))

        biassq.append(mean(biassq_temp))
    return biassq


def calculate_variance_rr(model_dict, D, y_true, x_true, L):
    variance = []
    for l in L:
        for s in x_true:
            y_pred = []
            var_temp = []
            for d in range(D):
                model = model_dict[l][d]
                y_pred.append(model.predict([gx(4, s)]).getT().tolist()[0])
            y_pred_exp = mean(y_pred)
            var_temp.append(
                sum(np.power(np.subtract(np.array(y_pred), np.array(np.repeat(y_pred_exp, D))), 2)) / (D - 1))
        variance.append(mean(var_temp))
    return variance


def main():
    #######################################################################################################################
    # Bias - Variance Trade-Off

    ##########################################################################
    # Part a
    # Generating datasets - 10 samples, 100 datasets
    print "Linear Regression with 10 samples, 100 datasets"
    datasets = gen_dataset(10, 100)

    # Applying LR on g's and calculating y's and w's
    results = train_lr(datasets)
    model_dict = results['model']
    sse_dict = results['sse']
    mse_dict = results['mse']
    y_predict_dict = results['y_predict_dict']

    # Plotting the histograms
    for g in range(1, 7):
        g_x = "g({g_})".format(g_=g)
        plothist(mse_dict[g].values(), g_x, 10)

    # Calculating Bias and Variance
    temp_dataset = gen_dataset(1, 100)
    y_true = [y for y in reduce(lambda y1, y2: y1 + y2, [zip(*dataset)[1] for dataset in temp_dataset])]
    x_true = [x for x in reduce(lambda x1, x2: x1 + x2, [zip(*dataset)[0] for dataset in temp_dataset])]
    biassq = calculate_biassq_lr(y_predict_dict, datasets, 10, 100)
    variance = calculate_variance_lr(model_dict, 100, y_true, x_true)

    for g, b in enumerate(biassq):
        print 'Bias Square for g({g}) = {b}'.format(g=g + 1, b=b)

    print '\n'
    for g, v in enumerate(variance):
        print 'Variance for g({g}) = {v}'.format(g=g + 1, v=v)

    ###########################################################################
    # Part b
    # Generating datasets  - 100 samples, 100 datasets
    print "\n\nLinear Regression with 100 samples, 100 datasets"
    datasets = gen_dataset(100, 100)

    # Applying LR on g's and calculating y's and w's
    results = train_lr(datasets)
    model_dict = results['model']
    sse = results['sse']
    mse = results['mse']
    y_predict_dict = results['y_predict_dict']

    # Plotting the histograms
    for g in range(1, 7):
        g_x = "g({g_})".format(g_=g)
        plothist(mse_dict[g].values(), g_x, 100)

    # Calculating Bias and Variance
    temp_dataset = gen_dataset(1, 100)
    y_true = [y for y in reduce(lambda y1, y2: y1 + y2, [zip(*dataset)[1] for dataset in temp_dataset])]
    x_true = [x for x in reduce(lambda x1, x2: x1 + x2, [zip(*dataset)[0] for dataset in temp_dataset])]
    biassq = calculate_biassq_lr(y_predict_dict, datasets, 100, 100)
    variance = calculate_variance_lr(model_dict, 100, y_true, x_true)

    for g, b in enumerate(biassq):
        print 'Bias Square for g({g}) = {b}'.format(g=g + 1, b=b)

    print '\n'

    for g, v in enumerate(variance):
        print 'Variance for g({g}) = {v}'.format(g=g + 1, v=v)

    ###########################################################################
    # Part d
    print "\n\nRegularized Linear Regression with 100 samples, 100 datasets"

    datasets = gen_dataset(100, 100)
    L = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    results = train_rr(datasets, L)
    y_predict_dict = results['y_predict_dict']
    model_dict = results['model']

    temp_dataset = gen_dataset(1, 100)
    y_true = [y for y in reduce(lambda y1, y2: y1 + y2, [zip(*dataset)[1] for dataset in temp_dataset])]
    x_true = [x for x in reduce(lambda x1, x2: x1 + x2, [zip(*dataset)[0] for dataset in temp_dataset])]
    biassq = calculate_biassq_rr(y_predict_dict, datasets, 100, 100, L)
    variance = calculate_variance_rr(model_dict, 100, y_true, x_true, L)

    for i, b in enumerate(biassq):
        print 'Bias Square for lambda = {l} = {b}'.format(l=L[i], b=b)

    print '\n'

    for i, v in enumerate(variance):
        print 'Variance for lambda = {l} = {v}'.format(l=L[i], v=v)

    #######################################################################################################################
    # Linear Kernel and SVM

    ##########################################################################
    # Loading the dataset
    train = scipy.io.loadmat('phishing-train.mat')
    test = scipy.io.loadmat('phishing-test.mat')
    x_train = train['features']
    y_train = train['label']
    x_test = test['features']
    y_test = test['label']

    ##########################################################################
    # Data pre-processing
    x_train_tf = feature_transform(x_train)
    x_test_tf = feature_transform(x_test)
    y_train = y_train.tolist()[0]
    y_test = y_test.tolist()[0]

    ##########################################################################
    # Use linear SVM in LIBSVM
    # Calculating best parameter for linear svm
    print "\n\nTraining SVM using Linear kernel"
    C = [c for c in get_param(math.pow(4, -6), math.pow(4, 2), 4)]
    svm_dict = {}
    for i, c in enumerate(C):
        start_time = time.time()
        model = svm_train(y_train, x_train_tf, ["-v", 3, "-c", c, "-t", 0, "-q"])
        svm_dict[i] = (model, (time.time() - start_time) / 3.0)
        print "C = {c:.6f}, time =  {t:.3f}, accuracy =  {a:.2f}".format(c=C[i], t=svm_dict[i][1], a=svm_dict[i][0])

    i = max(svm_dict.iterkeys(), key=(lambda key: svm_dict[key][0]))
    best_c_linear = C[i]
    best_accuracy_linear, training_time_linear = svm_dict[i]

    ##########################################################################
    # Kernel SVM in LIBSVM
    # Part a : Polynomial kernel
    # Calculating best parameters for polynomial kernel

    print "\n\nTraining SVM using Polynomial kernel"

    C = [c for c in get_param(math.pow(4, -3), math.pow(4, 7), 4)]
    svm_dict = {}
    for i, c in enumerate(C):
        svm_dict[i] = {}
        for d in [1, 2, 3]:
            start_time = time.time()
            model = svm_train(y_train, x_train_tf, ["-v", 3, "-c", c, "-t", 1, "-d", d, "-q"])
            svm_dict[i][d] = (model, (time.time() - start_time) / 3.0)
            print "C = {c:.6f}, degree = {d}, time = {t:.3f}, accuracy = {a:.2f}".format(c=C[i], d=d,
                                                                                         t=svm_dict[i][d][1],
                                                                                         a=svm_dict[i][d][0])

    best_accuracy_poly = 0
    for i, dict in svm_dict.items():
        for d, tup in dict.items():
            if tup[0] > best_accuracy_poly:
                best_c_poly = C[i]
                best_d_poly = d
                best_accuracy_poly = tup[0]
                training_time_poly = tup[1]

    # Part b : RBF Kernel
    # Calculating best parameters for gaussian kernel

    print "\n\nTraining SVM using RBF kernel"
    C = [c for c in get_param(math.pow(4, -3), math.pow(4, 7), 4)]
    gamma = [g for g in get_param(math.pow(4, -7), math.pow(4, -1), 4)]
    svm_dict = {}
    for i, c in enumerate(C):
        svm_dict[i] = {}
        for j, g in enumerate(gamma):
            start_time = time.time()
            model = svm_train(y_train, x_train_tf, ["-v", 3, "-c", c, "-t", 2, "-g", g, "-q"])
            svm_dict[i][j] = (model, (time.time() - start_time) / 3.0)
            print "C = {c:.6f}, gamma = {g:.6f}, time = {t:.3f}, accuracy = {a:.2f}".format(c=C[i], g=g,
                                                                                            t=svm_dict[i][j][1],
                                                                                            a=svm_dict[i][j][0])

    best_accuracy_rbf = 0
    for i, dict in svm_dict.items():
        for j, tup in dict.items():
            if tup[0] > best_accuracy_rbf:
                best_c_rbf = C[i]
                best_g_rbf = gamma[j]
                best_accuracy_rbf = tup[0]
                training_time_rbf = tup[1]

    print "\n\nBest accuracy for SVM using Polynomial kernel = {a} for C = {c}, d = {d}".format(a=best_accuracy_poly,
                                                                                                c=best_c_poly,
                                                                                                d=best_d_poly)
    print "Best accuracy for SVM using RBF kernel = {a} for C = {c}, g= {g}".format(a=best_accuracy_rbf, c=best_c_rbf,
                                                                                    g=best_g_rbf)

    #####################################################################
    # Running SVM on the best parameters

    print "\n\nBased on the above results, RBF kernel is getting highest accuracy"
    print "Running SVM on test data for C = {c} and g = {g}".format(c=best_c_rbf, g=best_g_rbf)

    svm_results = {}
    # SVM with RBF kernel
    p_labs, p_acc, p_vals = run_svm(y_train, x_train_tf, y_test, x_test_tf, c=best_c_rbf, g=best_g_rbf, kernel='rbf')
    svm_results['rbf'] = (p_labs, p_acc, p_vals)


if __name__ == '__main__':
    main()
