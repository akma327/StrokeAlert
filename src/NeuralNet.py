import theano
from theano import tensor as T
import numpy as np
# import matplotlib.pyplot as pl
import random
from sklearn.decomposition import PCA

'''
Neural Network 
'''
def process_data(input_file):
	"""
	Parse the complete stroke trial database and retain relevant columns. 
	Data cleaning phase
	"""
	f = open(input_file, 'r')
	header = f.readline().split(",")
	data = []
	for line in f:
		data.append(map(float, line.split(",")))
	return header, np.array(data)

#here you have the relative directory of your data
#x being the learning features
#y being the outcome you're measuring

x_file = "../data/dead_or_alive/x_balanced.csv"
y_file = "../data/dead_or_alive/y_balanced_two_col.csv"
x_header, x_data = process_data(x_file)
y_header, y_data = process_data(y_file)

#initialize variables
PARTITION_RATIO = 0.8 # Percent of the entire data set to treat as training data 
HU1 = 5 #number of hidden units, in general should not exceed number of learning features or it will overfit. If HU=1 then the network effectively becomes logistic regression
SCALE = 0.01 #no need to change this
LAMBDA = 0.01 #regularization constant, increase to reduce overfitting. I usually set this between 0.001 to 0.1 
LR = 0.1 #learning rate constant
REPS = 20 #number of times you want to reset and rerun the network 
EPOCH = 1000 #number of learning epochs. I set this network up as mini-batch gradient descent. Each mini-batch contains 128 data points. A learning epoch involves the network going through the entire learning dataset in mini-batches of 128.

#Initialize neural network methods and variables

def floatX(X):
    return np.asarray(X, dtype = theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape)*SCALE))

def rectify(X):
    return T.maximum(X, 0.)
        
#generalize to compute gradient descent on all model parameters
def sgd(cost, params, lr=LR):
    grads = T.grad(cost=cost, wrt = params)
    updates=[]
    for p,g in zip(params, grads):
        updates.append([p, p-g*lr])
    return updates
    
def Regularizer(w_h, w_o, lambda_l1=LAMBDA):
        return (lambda_l1/2)*(abs(w_h).mean() + abs(w_o).mean())
        
#2 layers of computation
#input --> hidden (sigmoid)
#hidden --> output (softmax)
def model(X, w_h, w_o, b, b2):
	h = rectify(T.dot(X, w_h) + b)
	pyx = T.nnet.softmax(T.dot(h, w_o) + b2)
	return pyx

X = T.fmatrix()
Y = T.fmatrix()

cc_all = []
acc_all = []


for r in range(REPS):
      
    '''
    shuffle everything!!
    '''
    temp = list(zip(x_data, y_data))
    random.shuffle(temp)
    x_data_r, y_data_r = zip(*temp)
    x_data = np.asarray(x_data_r)
    y_data = np.asarray(y_data_r)

    x_partition_idx = int(len(x_data)*PARTITION_RATIO)
    y_partition_idx = int(len(y_data)*PARTITION_RATIO)

    trX = x_data[:x_partition_idx]
    teX = x_data[x_partition_idx:]
    trY = y_data[:y_partition_idx]
    teY = y_data[y_partition_idx:]

    
    #Initialize both weight matrices
    w_h = init_weights((trX.shape[1], HU1))
    w_o = init_weights((HU1,2))
    b = init_weights((HU1,))
    b2 = init_weights((2,))

    py_x = model(X, w_h, w_o, b, b2)            
    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y)) + Regularizer(w_h, w_o, lambda_l1=LAMBDA)
    params = [w_h, w_o, b, b2]
    updates = sgd(cost,params) 

    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

    cc=[]
    acc = []
    
    for i in range(EPOCH):
        for start, end in zip(range(0,len(trX),128), range(128,len(trX),128)):
            cost = train(trX[start:end], trY[start:end]) 
        if(i % 50 == 0):
            prediction_accuracy = np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX), axis=1))
            print("REP: %s EPOCH: %s COST: %s PRED_ACCURACY: %s" % (r, i, cost, prediction_accuracy))
        cc.append(cost)
        acc.append(np.mean(np.argmax(teY, axis=1) == np.argmax(predict(teX), axis=1)))
        

    cc_all.append(cc)
    acc_all.append(acc)

    np.set_printoptions(precision = 3, linewidth = 10000)  

cc_all = np.asarray(cc_all) #in the end, cc_all contains the cost (or error) after each learning epoch, for all REP repetitions. You can plot this out.
acc_all = np.asarray(acc_all) #similarly, acc_all contains the final accuracy after each REP run of the network. 
  
print "**********SUMMARY************************"

print "Hidden units: ", HU1
print "Regularization constant (lambda): ", LAMBDA
print "Number of repetitions: ", REPS
print "Mean Classification Accuracy: ", np.mean(acc_all), " (SD = ", np.std(acc_all), ")"
print "Max Accuracy: ", np.max(acc_all)

print "**********END****************************"
print "To see classification accuracy of all runs, type 'acc_all'. To plot the cost function of all runs, type '%pylab' in iPython then type 'pl.plot(cc_all.transpose()). To plot the cost function of a particular run, type 'pl.plot(cc_all.transpose()[run_number])'" 
