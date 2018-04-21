import numpy as np
import math
from tqdm import tnrange, tqdm_notebook
from sklearn.preprocessing import LabelBinarizer

class logistic_model_scratch:
    
    def __init__(self, train_x, train_y, classes, data_dir, batch_size = 64):
        
        self.train_x = train_x
        self.train_y = train_y
        self.classes = classes
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.loss = []
        batches = batch_gen(train_x, train_y, classes, data_dir, batch_size=1)
        batch, _ = next(batches)
        self.x_size = batch.shape[0]
        self.B, self.W = self.init_parameters()

    def init_parameters(self):
        self.W = np.zeros([self.x_size, len(self.classes)]) #np.random.randn(x_size,1)
        self.B = 0
        return self.B, self.W
    
    def forward_prop(self, x):
        return sigmoid(np.dot(self.W.T, x) + self.B)
    
    def back_prop(self, yhat, Y, batch, learn_rate):
        dz = (1/self.batch_size) * (yhat-Y.T)
        dW = np.dot(batch, dz.T)
        dB = np.sum(dz)
        self.W -= learn_rate*dW
        self.B -= learn_rate*dB
    
    def calc_log_loss(self, yhat, Y):
        loss = (-1/Y.shape[0])*np.sum(np.add(Y.T * np.log(yhat), (1-Y.T) * np.log(1-yhat)))
#         from IPython.core.debugger import Tracer; Tracer()()
        if math.isinf(loss) | np.isnan(loss):
            loss = np.nan
        return loss
    
    def predict(self, test_x, test_y = None):
        yhat = self.forward_prop(test_x)
        predictions = np.argmax(Y, axis = 1)
        if test_y is not None:
            loss = self.calc_log_loss(yhat, test_y)
            acc = np.sum(1 - (test_y - yhat))/test_y.shape[0]
        return predictions, acc
    
    def fit_batch(self, batch, Y, learn_rate, n = 0):
        yhat = self.forward_prop(batch)
        self.back_prop(yhat, Y, batch, learn_rate)
        if n % 5 == 0:
            self.loss.append(self.calc_log_loss(yhat, Y))
        if n % 15 == 0:
            print('iter ' + str(n) + ': mean loss = ' + str(np.nanmean(self.loss[-3:])))
        
    def fit(self, epochs, learn_rate, batch_size = None):
        
        if batch_size is not None:
            self.batch_size = batch_size
            
        batches = batch_gen(self.train_x, self.train_y, self.classes, self.data_dir, batch_size = self.batch_size)
        
        for ep in tnrange(epochs):
            print('epoch: ' + str(ep))
            [self.fit_batch(batch, Y, learn_rate/(1+.05*num), n=num) for num, [batch, Y] in tqdm_notebook(enumerate(batches))]

            batches = batch_gen(self.train_x, self.train_y, self.classes, self.data_dir, batch_size = batch_size)
    