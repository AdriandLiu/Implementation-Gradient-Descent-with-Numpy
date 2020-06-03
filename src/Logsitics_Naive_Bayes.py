import numpy as np
import pandas as pd
import time
data = pd.read_csv("C:/Users/Donghan/Desktop/551A1/adult.data", delimiter=",", header = None)

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]


class Logistic:
    def __init__(self, learning_rate, n_iter, intercept = True):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.intercept = intercept

    # Sigmoid function provided by slides
    # y_h
    def sigmoid(self, X, weight):
        z = np.dot(X, weight)
        return 1 / (1 + np.exp(-z))

    def cost(self, weight, X, y):
        # h = np.dot(X, weight)
        # cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
            # Computes the cost function for all the training samples
        m = X.shape[0]
        cost = -(1 / m) * np.sum(y * np.log(self.sigmoid(X, weight)) + (1 - y) * np.log(
                1 - self.sigmoid(X,weight)))
        return cost

    def gradientDescent(self, X, y_h, y, weight):
        gradient = np.dot(X.T, (y_h - y)) / len(X)
        weight -= self.learning_rate * gradient
        return weight

    # CrossEntropyLoss formula provided by slides
    # def loss(self, y_h, y):
    #     loss = np.mean((-y * np.log(y_h) - (1 - y) * np.log(1 - y_h)))
    #     return loss

    def fit(self, X, y):
        if self.intercept == True:
            X = np.concatenate((np.ones((X.shape[0], 1)),X), axis = 1)

        # Weights
        self.weight = np.zeros(X.shape[1])
        # y_hat = self.sigmoid(X, theta)
        acc = []
        loss = []
        for i in range(self.n_iter):
            y_h = self.sigmoid(X, self.weight)
            # print((y_h.shape,y.shape))
            self.weight = self.gradientDescent(X, y_h, y, self.weight)
            acc.append(np.mean((self.sigmoid(X, self.weight)>=0.5)==y))
            # print("loss: {}".format(self.cost(self.weight, X,y)))
            loss.append(self.cost(self.weight, X, y))
    def predict(self, X):
        if self.intercept == True:
            X = np.concatenate((np.ones((X.shape[0], 1)),X), axis = 1)
        return self.sigmoid(X, self.weight) >= 0.5



class naiveBayes:
        def __init__(self):
            return None

        def BernoulliNaiveBayes(self, prior, likelihood, X,
            ):
            log_p = np.log(prior) + np.sum(np.log(likelihood) * X[:,None], 0) + \
            np.sum(np.log(1-likelihood) * (1 - X[:,None]), 0)
            log_p -= np.max(log_p) #numerical stability
            posterior = np.exp(log_p) # vector of size 2
            posterior /= np.sum(posterior) # normalize
            return posterior # posterior class probability

        def MultinomialNaiveBayes(self, X, y):
            '''
            calculate every P(...|class) by freq
            '''

            # categorical = [i for i in range(0,len(X_test.columns)) if X_test.iloc[:,i].dtype == "object"]
            prior = pd.DataFrame(X).groupby(y).agg(lambda x: len(x) / len(X)).iloc[:,0].astype('float64') # Income Prior
            # likelihood for each p(...|class), formula from slides.
            likelihood = [pd.DataFrame(y).groupby([y, X.iloc[:,i]]).agg(lambda x: len(x)).astype('float64').iloc[:,0] for i in range(len(self.categorical))]
            log_likelihood = likelihood.copy()

            for i in range(len(self.categorical)):
                for j in np.unique(y):
                    log_likelihood[i][j] = np.log(pd.DataFrame(X).groupby(y).agg(lambda x: len(x) / len(X)).iloc[:,0]\
                    .astype('float64')[j]+(likelihood[i][j]/pd.DataFrame(y).groupby(y).count().iloc[:,0][j])) # prior + likelihood
            log_likelihood_reordered = [i.reorder_levels([1,0]) for i in log_likelihood] # for future easy indexing use
            return log_likelihood_reordered


        def GaussianNaiveBayes(self, X, # N x D
            y):
            # From slides
            X = X.values
            y = pd.get_dummies(y).values
            N,C = y.shape
            D = X.shape[1]
            self.mu, self.s = np.zeros((C,D)), np.zeros((C,D))
            for c in range(C): #calculate mean and std
                inds = np.nonzero(y[:,c])[0]
                self.mu[c,:] = np.mean(X[inds,:], 0)
                self.s[c,:] = np.std(X[inds,:], 0)
                self.log_prior = np.log(np.mean(y, 0))[:,None]

            return self.mu, self.s, self.log_prior


        def fit(self, X, y):
            '''
            Prepare parameters, calculate from previous functions, based on the training data, namely, freq calculated based upon training set
            '''
            self.X = pd.DataFrame(X)
            self.y = y
            if len(np.unique(X.dtypes))<=0:
                return None

            if len(np.unique(X.dtypes))>=1:
                self.categorical = [i for i in range(0,len(X.columns)) if X.iloc[:,i].dtype == "object"] # categorical features
                self.continous = [i for i in range(0,len(X.columns)) if X.iloc[:,i].dtype == "int64" or X.iloc[:,i].dtype == "float64"]
                X_cate = X.iloc[:,self.categorical]
                X_cont = X.iloc[:,self.continous]
                if len(X_cate.columns) + len(X_cont.columns) != len(X.columns):
                    print("Mannually choose categorical and continous features")
                    return
                if not X_cont.empty:
                    self.mu, self.s, self.log_prior = self.GaussianNaiveBayes(X_cont, self.y)
        def predict(self, Xtest):
            '''
            After fitting, we have training set's parameters handy, now, use test set to make the prediction in accordance with the likelihood obtained above
            '''
            Xtest = pd.DataFrame(Xtest)
            X_test_cate = Xtest.iloc[:,self.categorical]
            X_test_cont = Xtest.iloc[:,self.continous]
            X_cate = self.X.iloc[:,self.categorical]
            X_cont = self.X.iloc[:,self.continous]
            multinomial = np.zeros((len(X_test_cate), len(np.unique(self.y))))
            gaussian = np.zeros((len(np.unique(y)),len(X_test_cate)))
            if not X_cate.empty:
                log_likelihood_reordered = self.MultinomialNaiveBayes(X_cate, self.y) # get log likelihood table from training set
                for i in range(len(X_test_cate)):
                    for j in range(len(X_test_cate.columns)):
                        multinomial[i] += np.array(log_likelihood_reordered[j][X_test_cate.values[i][j]]) # log likelihood table from test set N*(n of classes)
            if not X_cont.empty:
                if np.isnan(np.log(self.s[:,None,:])).any():
                    print("Check data input for 'zero', leads to log(0) => nan")
                log_likelihood = - np.sum(np.log(self.s[:,None,:]) +.5*(((X_test_cont.values[None,:,:]- self.mu[:,None,:])/self.s[:,None,:])**2), 2) # from lecture, gaussian log likelihood
                gaussian = self.log_prior + log_likelihood
            predictions = np.argmax(multinomial + gaussian.T, axis = 1) # combine gaussian and multinomial in mix type of features data, get the max
            return predictions

def mean_normal(data):
    # Mean normalization
    return (data - data.mean())/ data.std()

def min_max_normal(data):
    # Min-max normalization
    return (data-data.min())/(data.max()-data.min())

def evaluate_acc(y_h, y):
    # self-correction
    if np.mean(y_h == y) < 0.5:
        # print(y_h, y)
        acc = 1 - np.mean(y_h == y)
    return np.mean(y_h == y)

def cross_validation_train(fit, X, y, k, learning_rate = None, n_iter = None, intercept = True):
    '''CV training, with k-folder and k times of training'''
    if fit == "logistic":
        model = Logistic(learning_rate = learning_rate, n_iter = n_iter, intercept = intercept)
    else:
        model = naiveBayes()
    training_accuracy = []
    validation_accuracy = []
    start = list(range(len(X.index)))
    i = 0
    while i<k :
        try:
            index = np.random.choice(start, replace = False, size = int((len(X.index))*((k-1)/k)))
            # Avoid duplicate
            val_index = list(set(start) - set(index))
            # Training
            model.fit(X.iloc[index], y[index])
            # validation accuracy
            # Choose the one portion of the k-folded data
            validation_accuracy.append(evaluate_acc(model.predict(X.iloc[val_index]), y[val_index]))
            # Training accuracy
            training_accuracy.append(evaluate_acc(model.predict(X.iloc[index]), y[index]))
            i += 1
        except KeyError:
            continue
    return validation_accuracy, training_accuracy, model


def precision_score(y_pred, y_true):
    return ((y_true==1)*(y_pred==1)).sum()/(y_pred==1).sum()
def recall_score(y_pred, y_true):
    return ((y_true==1)*(y_pred==1)).sum()/(y_true==1).sum()
def f1_score(y_pred, y_true):
    num = 2*precision_score(y_true, y_pred)*recall_score(y_true, y_pred)
    deno = (precision_score(y_true, y_pred)+recall_score(y_true, y_pred))
    return num/deno

def confusion_matrix(y_h, y):
    # Confusion matrix
    return pd.crosstab(pd.Series(y_h, name = 'prediction'), pd.Series(y, name = "true"))
if __name__ == '__main__':

    X_adult = pd.get_dummies(X) # Char / calegratial to one-hot encoding
    y_adult = np.asarray([0 if i == " <=50K" else 1 for i in y]) # Char to numerical
    y_ionosphere = np.array([0 if i == "b" else 1 for i in y])
    y_wine = np.array([0 if i == 3 or i == 4 or i == 5 else 1 for i in y])
    y_wdbc = np.array([0 if i == "M" else 1 for i in y])

    # Cross Validation
    k = 5
    # NB is slower than Logistics

    # Higher accuracy with application of normalization
    # say, mean_normal(X) or min_max_normal(X)
    start = time.time()
    log_validation_accuracy, log_training_accuracy, model_log = cross_validation_train("logistic",mean_normal(X_adult.iloc[:,:]), y_adult, k, learning_rate = 0.01, n_iter = 2000, intercept = True)
    time.time() - start
    start = time.time()
    nb_validation_accuracy, nb_training_accuracy, model_nb = cross_validation_train("NB",X, y, k)
    time.time() - start
    confusion_matrix([1 if i == True else 0 for i in model_log.predict(mean_normal((X_adult.iloc[:,:])))], y_adult)
    # data = pd.read_csv("C:/Users/Donghan/Desktop/551A1/winequality-red.data", delimiter=";")
    # Recall, precision, F-1 score
    def eval(model, X, y):
        return precision_score(np.array([1 if i == True else 0 for i in model.predict(mean_normal(pd.get_dummies(X)))]), y),recall_score(np.array([1 if i == True else 0 for i in model.predict(mean_normal(pd.get_dummies(X)))]), y),f1_score(np.array([1 if i == True else 0 for i in model.predict(mean_normal(pd.get_dummies(X)))]), y)

    eval(model_log, X_adult.iloc[:,:], y_adult)



    # Feature importance for wine data
    dict(zip(model_log.weight,["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]))


    data = data.drop(["citric acid", "density","pH"], axis = 1)
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]



    print("Logistic Validation Accuracy: {}".format(np.mean(log_validation_accuracy)))
