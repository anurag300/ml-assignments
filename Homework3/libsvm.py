from svmutil import *


def run_svm(y_train, x_train, y_test, x_test, **option):
    if not y_train or not y_test or not x_test or not x_train:
        raise Exception("Please provide proper data")

    kernel = option.pop('kernel', None)

    if kernel == None:
        c = option.pop('c', None)
        if c == None:
            raise ValueError("Please specify parameters for linear SVM")

        model = svm_train(y_train, x_train, ["-c", c, "-t", 0, "-q"])
        p_labs, p_acc, p_vals = svm_predict(y_test, x_test, model)
        return p_labs, p_acc, p_vals

    elif kernel == 'poly':
        c = option.pop('c', None)
        d = option.pop('d', None)
        if c == None or d == None:
            raise ValueError("Please specify kernel parameters for Polynomial kernel")

        model = svm_train(y_train, x_train, ["-c", c, "-t", 1, "-d", d, "-q"])
        p_labs, p_acc, p_vals = svm_predict(y_test, x_test, model)
        return p_labs, p_acc, p_vals


    elif kernel == 'rbf':
        c = option.pop('c', None)
        g = option.pop('g', None)
        if c == None or g == None:
            raise ValueError("Please specify kernel parameters for RBF kernel")

        model = svm_train(y_train, x_train, ["-c", c, "-t", 2, "-g", g, "-q"])
        p_labs, p_acc, p_vals = svm_predict(y_test, x_test, model)
        return p_labs, p_acc, p_vals

    else:
        raise Exception("Unknown parameter type")
