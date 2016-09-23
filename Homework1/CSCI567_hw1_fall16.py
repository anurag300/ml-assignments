import knn
import naivebayes as nb



def run_naive_bayes():
    print "              RUNNING NAIVE BAYES CLASSIFIER"
    print "######################################################"
    with open('train.txt') as f:
        train_data = [line.split()[0].split(',') for line in f.readlines()]

    train_data = map(lambda data_rec : map(lambda x : float(x), data_rec), train_data)
    train_data = [data[1:-1] + [data[-1]] for data in train_data]

    Y_train = []
    for data in train_data:
        Y_train.append(data[-1:][0])

    #calculating probabilities of each class
    classProbDict = nb.calculateClassProb(Y_train)


    #storing training data as -- class:(upvote,feature vector)
    train_data_map = {}
    for data in train_data:
        if data[-1:][0] in train_data_map:
            train_data_map[data[-1:][0]].append(data[:-1])
        else:
            train_data_map[data[-1:][0]] = []
            train_data_map[data[-1:][0]].append(data[:-1])

    #calculating the parameters (sigma, mu) for each class
    train_data_summmary = {}
    for key, val in train_data_map.items():
        train_data_summmary[key] = [nb.getSummary(data) for data in zip(*val)]


    with open('test.txt') as f:
        test_data = [line.split()[0].split(',') for line in f.readlines()]
    test_data = map(lambda data_rec: map(lambda x: float(x), data_rec), test_data)
    test_data = [data[1:-1] + [data[-1]] for data in test_data]


    Y_test = [data[-1:][0] for data in test_data]
    Y_predict_test = []
    for input in test_data:
        test_data_map = {}
        for key, val in train_data_summmary.items():
            test_data_map[key] = nb.calculateNaiveBayesProb(input[:-1],val,key,classProbDict)
        Y_predict_test.append(max(test_data_map, key = test_data_map.get))


    Y_predict_train = []
    for input in train_data:
        train_data_dict = {}
        for key, val in train_data_summmary.items():
            train_data_dict[key] = nb.calculateNaiveBayesProb(input[:-1], val,key,classProbDict)
        Y_predict_train.append(max(train_data_dict, key=train_data_dict.get))

    #print "Given Test Classes    ", Y_test
    #print "Predicted Test Classes", Y_predict_test
    result_test = [test == predicted for test, predicted in zip(Y_test, Y_predict_test)]
    print "Test Accuracy = {:.2f}".format(result_test.count(True) / float(len(result_test)) * 100)

    #print "Given Training Classes    ", Y_train
    #print "Predicted Training Classes", Y_predict_train
    result_train = [test == predicted for test, predicted in zip(Y_train, Y_predict_train)]
    print "Train Accuracy = {:.2f}".format(result_train.count(True) / float(len(result_train)) * 100)
    print "\n"


def run_knn():
    print "              RUNNING KNN CLASSIFIER"
    print "######################################################"

    with open('train.txt') as f:
        train_data = [line.split()[0].split(',') for line in f.readlines()]
    train_data = map(lambda data_rec: map(lambda x: float(x), data_rec), train_data)
    train_data = [data[1:-1] + [data[-1]] for data in train_data]
    data_stats = knn.getDataStats([train_data_inst[:-1] for train_data_inst in train_data])
    train_data = [knn.standardize(train_data_inst[:-1],data_stats) + [train_data_inst[-1]] for train_data_inst in train_data]
    Y_train = [data[-1:][0] for data in train_data]

    with open('test.txt') as f:
        test_data = [line.split()[0].split(',') for line in f.readlines()]
    test_data = map(lambda data_rec: map(lambda x: float(x), data_rec), test_data)
    test_data = [data[1:-1] + [data[-1]] for data in test_data]
    test_data = [knn.standardize(test_data_inst[:-1],data_stats) + [test_data_inst[-1]] for test_data_inst in test_data]
    Y_test = [data[-1:][0] for data in test_data]
    test_data = [data[:-1] for data in test_data]


    # On test data
    print "         Results for test data"
    print "-----------------------------------------"
    print "            L1      |    L2"
    for k in [1,3,5,7]:
        Y_predictL1 = knn.knn(train_data, test_data,k, 'l1')
        Y_predictL2 = knn.knn(train_data, test_data,k, 'l2')
        resultL1 = [test == predicted for test, predicted in zip(Y_test, Y_predictL1)]
        resultL2 = [test == predicted for test, predicted in zip(Y_test, Y_predictL2)]
        #print "Predicted Classes",Y_predictL1
        #print "Given Classes    ",Y_test
        L1 = "{:.2f}".format(resultL1.count(True) / float(len(resultL1)) * 100)
        #print "Predicted Classes", Y_predictL2
        #print "Given Classes    ", Y_test
        L2 = "{:.2f}".format(resultL2.count(True) / float(len(resultL2)) * 100)
        print "K = {k}      {L1}    |   {L2}".format(k=k,L1=L1,L2=L2)
    print "\n"

    # On train data
    print "       Results for training data"
    print "----------------------------------------"
    print "            L1      |    L2"
    for k in [1, 3, 5, 7]:
        Y_predictL1 = []
        Y_predictL2 = []
        for i in xrange(len(train_data)):
            train_train_data = train_data[0:i]+train_data[i+1:]
            train_test_data = [train_data[i]]
            train_test_data = [data[:-1] for data in train_test_data]
            Y_predictL1.append(knn.knn(train_train_data, train_test_data, k, 'l1')[0])
            Y_predictL2.append(knn.knn(train_train_data, train_test_data, k, 'l2')[0])
        resultL1 = [test == predicted for test, predicted in zip(Y_train, Y_predictL1)]
        resultL2 = [test == predicted for test, predicted in zip(Y_train, Y_predictL2)]

        #print "Predicted Classes", Y_predictL1
        #print "Given Classes    ", Y_train
        L1 = "{:.2f}".format(resultL1.count(True) / float(len(resultL1)) * 100)
        # "Predicted Classes", Y_predictL2
        #print "Given Classes    ", Y_train
        L2 = "{:.2f}".format(resultL2.count(True) / float(len(resultL2)) * 100)
        print "K = {k}      {L1}    |   {L2}".format(k=k, L1=L1, L2=L2)



def main():
    run_naive_bayes()
    run_knn()

if __name__ == "__main__":
    main()