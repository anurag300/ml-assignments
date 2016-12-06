from hw_utils import *
import time

DATA_FILE = 'MiniBooNE_PID.txt'


def main():
    X_tr, y_tr, X_te, y_te = loaddata(DATA_FILE)
    X_tr_norm, X_te_norm = normalize(X_tr=X_tr, X_te=X_te)
    din = 50
    dout = 2

    ###################################################################################################################
    #Linear activations
    print 'Linear activations'
    start = time.time()
    archs = [[din,dout],[din,50,dout],[din,50,50,dout],[din,50,50,50,dout]]
    testmodels(X_tr_norm,y_tr,X_te_norm,y_te,archs=archs,actfn='linear',
               sgd_lr = 0.001, reg_coeffs = [0.0],num_epoch = 30,
               batch_size = 1000, sgd_decays = [0.0],sgd_Nesterov = False, EStop=False,verbose=0 )
    print 'Time take :',str(time.time()-start)

    archs = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout],
             [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
    start = time.time()
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='linear',
               sgd_lr = 0.001, reg_coeffs = [0.0],num_epoch = 30,
               batch_size = 1000, sgd_decays = [0.0],sgd_Nesterov = False, EStop=False,verbose=0  )
    print 'Time take :',str(time.time()-start)

    #Sigmoid activation
    print '\n\nSigmoid activation'
    archs = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout],
             [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
    start = time.time()
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='sigmoid',
               sgd_lr=0.001, reg_coeffs=[0.0], num_epoch=30,
               batch_size=1000, sgd_decays=[0.0], sgd_Nesterov=False, EStop=False,verbose=0
               )
    print 'Time take :', str(time.time() - start)

    #ReLu activation
    print '\n\nReLu activation'
    archs = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout],
             [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
    start = time.time()
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='relu', sgd_lr=5e-4,
               reg_coeffs=[0.0], num_epoch=30,
               batch_size=1000, sgd_decays=[0.0], sgd_Nesterov=False, EStop=False,verbose=0
               )
    print 'Time take :', str(time.time() - start)

    #L2-Regularization
    print '\n\nL2-Regularization'
    archs = [[din, 800, 500, 300, dout]]
    reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='relu', sgd_lr=5e-4,
               reg_coeffs=reg_coeffs,  num_epoch=30,batch_size=1000, sgd_decays=[0.0],
               sgd_Nesterov=False, EStop=False,verbose=0)

    #Early stopping and L2-Regularization
    print '\n\nEarly stopping anf L2-Regularization'
    archs = [[din, 800, 500, 300, dout]]
    reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='relu', sgd_lr=5e-4,
               reg_coeffs=reg_coeffs,  num_epoch=30,batch_size=1000, sgd_decays=[0.0],
               sgd_Nesterov=False, EStop=True,verbose=0)

    #SGD with weight decay
    print '\n\nSGD with weight decay'
    archs = [[din, 800, 500, 300, dout]]
    reg_coeffs = [5e-7]
    sgd_decays = [1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3]
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='relu', sgd_lr=1e-5,
               reg_coeffs=reg_coeffs, EStop=False,num_epoch=100,batch_size=1000,
               sgd_decays=sgd_decays,verbose=0)


    best_decay = 5e-05
    #Momentum
    print '\n\nMomentum'
    archs = [[din, 800, 500, 300, dout]]
    reg_coeffs = [0.0]
    sgd_moms = [0.99,0.98,0.95,0.9,0.85]
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='relu', sgd_lr=1e-5,
               reg_coeffs=reg_coeffs, EStop=False, num_epoch=50, batch_size=1000,
               sgd_decays=[best_decay],sgd_Nesterov=True,sgd_moms = sgd_moms,verbose=0)

    best_reg_coeff = [1e-05]
    best_mom = [0.99]
    best_decay = [5e-5]

    # Combining the above
    print '\n\nCombining the above'
    archs = [[din, 800, 500, 300, dout]]
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='relu', sgd_lr=1e-5,
               reg_coeffs=best_reg_coeff, EStop=True, num_epoch=100, batch_size=1000,
               sgd_decays=best_decay, sgd_Nesterov=True, sgd_moms=best_mom, verbose=0)

    # Grid search with cross validation
    print '\n\nGrid search with cross validation'
    archs = [[din, 50, dout], [din, 500, dout], [din, 500, 300, dout],
             [din, 800, 500, 300, dout], [din, 800, 800, 500, 300, dout]]
    reg_coeffs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    sgd_decays = [1e-5, 5e-5, 1e-4]
    testmodels(X_tr_norm, y_tr, X_te_norm, y_te, archs=archs, actfn='relu', sgd_lr=1e-5,
               reg_coeffs=reg_coeffs, EStop=True, num_epoch=100, batch_size=1000,
               sgd_decays=sgd_decays, sgd_Nesterov=True, sgd_moms=[0.99], verbose=0)


if __name__ == '__main__':
    main()
