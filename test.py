import numpy as np
import matplotlib.pyplot as plt
from model import MDN, DensityEstimator
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
    result = []
    for i in range(10):
        print("Run %d"%i)
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
        result.append(model)

    best = result[np.argmin([m(xi.reshape(-1,1),yi).data for m in result])]
    serializers.save_npz("result/model.npz", best)

    plot(best.predictor, xi, yi)

def plot(model, xi, yi):
    X,Y = np.meshgrid(np.linspace(-2,2), np.linspace(-1,2))
    print(X.reshape(-1,1).astype(np.float32).shape)
    print(X[:,0].reshape(-1,1).astype(np.float32).shape)
    Z = model(X.reshape(-1,1).astype(np.float32),
              Y.reshape(-1,1).astype(np.float32)).data.reshape(*X.shape)
    plt.contourf(X,Y,Z)
    plt.plot(xi,yi, "w.")
    mean = model.mean(X[:,0].reshape(-1,1).astype(np.float32))
    plt.plot(X[:,0], mean[0])
    plt.show()

if __name__ == "__main__":
    test_MDN()
