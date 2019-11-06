# Evolutionary Ensemble Learning

from bnn import BNN
import numpy as np
from activations import softmax

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

    def variation(self):
        """ Sample a fresh and bigger population from current population.
        """
        new_population = []
        for i in range(self.l): # offspring size
            model = self.population[i % self.n_estimators] # select an estimator from the current population
            W = model.W
            W = W # TODO: mutation
            candidate = self.init_estimator(W=W)
            new_population.append(candidate)
        return new_population

    def selection(self, X, y, new_population):
        """ Select best models from population.
        """
        # TODO: boostrap? random batch?
        losses = np.array([model.loss(X, y) for model in new_population])
        best = losses.argsort()[-self.n_estimators-1:][::-1] # maybe slow
        return np.extract(best, new_population)

    def fit(self, X, y, epochs=1):
        """ Evolve to a new generation of estimators.
        """
        print(self.population)
        for _ in range(epochs):
            new_population = self.variation()
            self.population = self.selection(X, y, new_population)
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
    model = EEL(layers=[2, 4])
    X = [[1, 3], [0, -3]]
    y = [0, 1]
    print('input')
    print(X)
    print('labels')
    print(y)
    print('output')
    print(model.predict_proba(X))
    print(model.predict(X))
    model.fit(X, y, epochs=2)
    print('output')
    print(model.predict_proba(X))
    print(model.predict(X))
