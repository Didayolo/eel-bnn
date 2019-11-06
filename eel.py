# Evolutionary Ensemble Learning

from bnn import BNN

class EEL():
    def __init__(self, layers=[10, 10, 10], n_estimators=10):
        """ Evolutationary Ensemble Learning for Binary Neural Networks.
        """
        population = []
        for _ in range(len(n_estimators)):
            estimator = BNN(layers=layers)
            population.append(estimator)
        self.population = population

    def variation(self):
        """ Sample a fresh and bigger population from current population.
        """
        return population + population

    def selection(self, X, y, population):
        """ Select best models from population.
        """
        for model in population:
            model.loss(X, y)
        return population

    def fit(self, X, y, epochs=1):
        """ Evolve to a new generation of estimators.
        """
        for _ in range(len(epochs)):
            self.population = self.selection(self.variation())

    def predict(self, X, y):
        """ Population's members vote together to predict y from X.
        """
        for model in population:
            model.predict_proba(X, y)
        pass
