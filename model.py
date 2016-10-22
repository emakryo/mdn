import numpy as np
from chainer import training, optimizers, serializers, utils, datasets, iterators, report
from chainer import Variable, Link, Chain, ChainList
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

class MDN(Chain):
    """
    Mixure density network model.
    """
    def __init__(self, IN_DIM=1, HIDDEN_DIM=10, OUT_DIM=1, NUM_MIXTURE=3):
        self.IN_DIM = IN_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.OUT_DIM = OUT_DIM
        self.NUM_MIXTURE = NUM_MIXTURE
        super(MDN, self).__init__(
            l1_ = L.Linear(IN_DIM, HIDDEN_DIM),
            coef_ = L.Linear(HIDDEN_DIM, NUM_MIXTURE),
            mean_ = L.Linear(HIDDEN_DIM, NUM_MIXTURE*OUT_DIM),
            logvar_ = L.Linear(HIDDEN_DIM, NUM_MIXTURE)
        )

    def __call__(self, x, y):
        h = F.sigmoid(self.l1_(x))
        coef = F.softmax(self.coef_(h))
        mean = F.reshape(self.mean_(h), (-1,self.NUM_MIXTURE,self.OUT_DIM))
        logvar = self.logvar_(h)
        mean, y = F.broadcast(mean, F.reshape(y, (-1,1,self.OUT_DIM)))
        return F.sum(
            coef*F.exp(-0.5*F.sum((y-mean)**2, axis=2)*F.exp(-logvar))/
            ((2*np.pi*F.exp(logvar))**(0.5*self.OUT_DIM)),axis=1)

    def mean(self, x):
        h = F.sigmoid(self.l1_(x))
        return F.reshape(self.mean_(h), (-1,self.NUM_MIXTURE, self.OUT_DIM))

    def var(self, x):
        h = F.sigmoid(self.l1_(x))
        return F.exp(self.logvar_(h))

class DensityEstimator(Chain):
    """
    Evaluator model for density estimation model.
    """
    def __init__(self, predictor):
        super(DensityEstimator,self).__init__(predictor=predictor)

    def __call__(self, *args):
        density = self.predictor(*args)
        nll = -F.sum(F.log(density))
        report({'nll': nll}, self)
        return nll
