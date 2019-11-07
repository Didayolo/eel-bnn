# Evolutionary Ensemble Learning

from bnn import BNN
import numpy as np
from activations import softmax

def mutation(arr, p=0.2):
    """ Random mutations of array arr with probability p.
    """
    m = np.random.choice([True, False], size=arr.shape, p=[p, 1-p])
    return np.where(m, -np.sign(arr), arr)

class EEL():
    def __init__(self, layers=[10, 10, 10], n_estimators=10, l=None):
        """ Evolutationary Ensemble Learning for Binary Neural Networks.

            Method used by default: Post-Mortem Ensemble Creation.

            :param layers: List of integers representing the size of each layer.
                           The first is the input size and the last is the output size.
            :param n_estimators: Number of estimators of the ensemble method.
                                 This also the size of the parent population ("mu").
            :param l: The size of the offsprings during training ("lambda").
                      Must be bigger than n_estimators.
        """
        self.layers = layers
        self.n_estimators = n_estimators
        self.output_size = layers[-1]
        self.generation = 0
        if l is None:
            l = n_estimators * 2
        if l < n_estimators:
            raise Exception('lambda (offsprings size) must be bigger than n_estimators (parent population size).')
        self.l = l
        population = []
        for _ in range(n_estimators):
            estimator = self.init_estimator()
            population.append(estimator)
        self.population = population

    def init_estimator(self, W=None, B=None):
        return BNN(layers=self.layers, W=W, B=B)

    def variation(self, p=0.2, keep=False):
        """ Sample a fresh and bigger population from current population.

            :param p: Probability of mutation.
            :param keep: If True, keeps current population in the new generation.
        """
        new_population = []
        for i in range(self.l): # offspring size
            model = self.population[i % self.n_estimators] # select an estimator from the current population
            # weights variation
            W = model.W
            new_W = []
            for Wi in W:
                new_W.append(mutation(Wi, p=p)) # mutation
            # bias variation
            B = model.B
            new_B = []
            for Bi in B:
                new_B.append(mutation(Bi, p=p)) # mutation
            candidate = self.init_estimator(W=new_W, B=new_B)
            new_population.append(candidate)
        if keep:
            return new_population + self.population
        return new_population

    def selection(self, X, y, new_population, batch_size=250, replace=True, verbose=False):
        """ Select best models from population.

            :param X: np.ndarray of data
            :param y: np.ndarray of target corresponding to data
            :param new_population: The population to select individuals from.
            :param batch_size: Size of sub-samples of data to feed models with.
            :param replace: If True, sample batches with replacement.
            :param verbose: If True, display information about loss.
        """
        losses = []
        for model in new_population:
            # TODO: boostrap? random batch?
            idx = np.random.choice(np.arange(len(X)), batch_size, replace=replace)
            X_sampled = X[idx] # TODO: this is not efficient
            y_sampled = y[idx]
            losses.append(model.loss(X_sampled, y_sampled)) # compute losses
        losses = np.array(losses)
        best = losses.argsort()[:self.n_estimators+1] # select models with lowest losses
        if verbose:
            #print('loss before selection = {}'.format(np.sum(losses) / len(new_population))) # generation's loss # TODO: learning curve?
            print('loss = {}'.format(np.sum(np.extract(best, losses)) / self.n_estimators)) # selected generation's loss
        return list(np.extract(best, new_population)) # populations only of type ndarray to avoid too many castings?

    def fit(self, X, y, epochs=1, p=0.2, constant_p=True, keep=False, batch_size=250, replace=True, power_t=0.2, verbose=False):
        """ Evolve to a new generation of estimators.

            :param X: Data
            :param y: Target corresponding to data
            :param epochs: Number of learning epochs.
            :param p: Initial probability of mutation (kind of "learning rate") during variation phase.
            :param constant_p: If True, keeps p constant (the learning rate).
            :param keep: If True, the current population is kept during the variation phase.
            :param batch_size: Size of sub-samples of data to feed models with during selection phase.
            :param replace: If True, sample batches with replacement during selection phase.
            :param power_t: Factor for invscaling learning rate.
            :param verbose: If True, display information about loss and learning rate evolution.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        # TODO: clean dynamic learning rate
        # adaptive: keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing.
        p_init = p # intial value of learning rate
        for i in range(epochs):
            if not constant_p:
                # decrease mutation probability during training (learning rate)
                # invscaling:
                p = p_init / pow(i+1, power_t)
                if verbose:
                    print('p = {}'.format(p))
            new_population = self.variation(p=p, keep=keep)
            self.population = self.selection(X, y, new_population, batch_size=batch_size, replace=replace, verbose=verbose)
            self.generation += 1

    def predict_proba(self, X, soft=True, proba=True):
        """ Population's members vote together to predict y from X.

            :param soft: If True soft vote, else hard vote.
            :param proba: If True apply softmax to result (to have a total of 1).
        """
        if soft:
            # just sum the probabilities
            sum = self.population[0].predict_proba(X)
            for model in self.population[1:]:
                sum = np.add(sum, model.predict_proba(X))
        else:
            # one hot of each prediction
            sum = np.identity(self.output_size)[self.population[0].predict(X)]
            for model in self.population[1:]:
                sum = np.add(sum, np.identity(self.output_size)[model.predict(X)])
        if proba:
            return softmax(sum.T).T
        return sum

    def predict(self, X, soft=True):
        """ Predict X's target with a majority vote between (trained) estimators.

            :param X: np.ndarray of data
            :param soft: If True do a soft vote, else do a hard vote.
        """
        y_pred = self.predict_proba(X, soft=soft, proba=False)
        return np.argmax(y_pred, axis=1) # take the most voted label

if __name__ == "__main__":
    print('testing...')
    model = EEL(layers=[1, 2], n_estimators=1, l=5)
    X = np.array([[-10], [-5], [-12], [10], [1], [4]])
    y = np.array([0, 0, 0, 1, 1, 1])
    print('input')
    print(X)
    print('labels')
    print(y)
    print('output')
    print(model.predict_proba(X))
    print(model.predict(X))
    print('training...')
    model.fit(X, y, epochs=500, p=0.01, batch_size=6, replace=False, verbose=False)
    print('output')
    print(model.predict_proba(X))
    print(model.predict(X))
    print('validation labels')
    X_test = np.array([[-8], [-6], [13], [3]])
    y_test = np.array([0, 0, 1, 1])
    print(y_test)
    print('validation output')
    print(model.predict(X_test))
    print(model.predict(X_test, soft=False))
