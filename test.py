import numpy as np
import matplotlib.pyplot as plt
from mdn import MDN, DensityEstimator
import chainer
from chainer import optimizers, serializers, training, iterators, datasets
from chainer import Variable, Link, Chain, ChainList
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

def uni_modal(n=500):
    xi = np.random.rand(n)
    yi = np.sin(2*xi) + 0.2*np.random.randn(n)*np.cos(2*xi)**2
    return xi,yi

def multi_modal(n=500):
    yi = np.random.rand(n)
    xi = np.sin(2*np.pi*yi) + 0.2*np.random.rand(n)*(np.cos(2*np.pi*yi)+2)
    return xi,yi

def test_MDN():
    import matplotlib.pyplot as plt
    weight_decay = 0.0005
    xi, yi = multi_modal()
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)
    dataset = datasets.tuple_dataset.TupleDataset(xi.reshape(-1,1),yi)
    model= DensityEstimator(MDN(1,10,1,3))
    optimizer = optimizers.AdaGrad(lr=0.2)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    train_iter = iterators.SerialIterator(dataset, batch_size=100)
    #test_iter = iterators.SerialIterator(dataset, batch_size=100,
    #                                     repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (2000, 'epoch'))
    #trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/nll']))
    trainer.run()
    X,Y = np.meshgrid(np.linspace(-1,1.2), np.linspace(0,1))
    Z = model.predictor(X.reshape(-1,1).astype(np.float32),
                        Y.reshape(-1,1).astype(np.float32)).data.reshape(*X.shape)
    print(np.unravel_index(Z.argmin(),X.shape), Z.min())
    print(np.unravel_index(Z.argmax(),X.shape), Z.max())
    plt.contourf(X,Y,Z)
    plt.plot(xi,yi, "w.")
    plt.show()

if __name__ == "__main__":
    test_MDN()
