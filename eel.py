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
            W = model.W
            new_W = []
            for Wi in W:
                new_W.append(mutation(Wi, p=p)) # mutation
            candidate = self.init_estimator(W=new_W)
            new_population.append(candidate)
        if keep:
            return new_population + self.population
        return new_population

    def selection(self, X, y, new_population, batch_size=250, replace=True):
        """ Select best models from population.
        """
        losses = []
        for model in new_population:
            # TODO: boostrap? random batch?
            idx = np.random.choice(np.arange(len(X)), batch_size, replace=replace)
            X_sampled = X[idx] # TODO: do this computation less times
            y_sampled = y[idx]
            losses.append(model.loss(X_sampled, y_sampled))
        losses = np.array(losses)
        #print(np.sum(losses)) # generation's total loss # TODO: learning curve?
        best = losses.argsort()[:self.n_estimators+1] # select model with lowest losses
        print('loss = {}'.format(np.sum(np.extract(best, losses)) / self.n_estimators)) # selected generation's loss
        return list(np.extract(best, new_population)) # populations only of type ndarray to avoid too many castings?

    def fit(self, X, y, epochs=1, p=0.2, keep=False, batch_size=250, replace=True):
        """ Evolve to a new generation of estimators.
        """
        # TODO: clean dynamic learning rate
        decrease = 0.05
        for i in range(epochs):
            # decrease mutation probability during training (learning rate)
            if p > 0.001:
                p -= decrease / (i+1)
                print('p = {}'.format(p))
            new_population = self.variation(p=p, keep=keep)
            self.population = self.selection(X, y, new_population, batch_size=batch_size, replace=replace)
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
        y_pred = self.predict_proba(X, soft=soft, proba=False)
        return np.argmax(y_pred, axis=1) # take the most voted label

if __name__ == "__main__":
    model = EEL(layers=[1, 2], n_estimators=2)
    X = np.array([[-10], [10]])
    y = np.array([0, 1])
    print('input')
    print(X)
    print('labels')
    print(y)
    print('output')
    print(model.predict_proba(X))
    print(model.predict(X))
    model.fit(X, y, epochs=500, p=0.5, batch_size=2, replace=False)
    print('output')
    print(model.predict_proba(X))
    print(model.predict(X))
