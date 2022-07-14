---
layout: tech
title: Many Smaller GPUs - Elegant In-Model Distribution
description: 
toc: true
---

GPUs (and TPUs) are no doubt one of our most critical resources when running neural networks.

GPUs have strict capacities and limits as physical resources. When using servers with many smaller GPUs, we often ran into the inherent physical limitations of our equipment.

Laying out a possibly large model across many smaller GPUs has thus become a requirement for us. This blog presents a few basic steps in that direction.

The outlined model can also be used to effectively test more complex and custom GPU allocation strategies.

Our objective here is to arrive at training a model representable by the [graph](generated/images/tech/gpus.pdf) and [runnable example](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/gpus.ipynb).

Just as before, we need to prep our environment to run any meaningful code:

```python
import numpy as np
import tensorflow as tf
from datetime import datetime
```

We also define a few convenient aliases:

```python
ks = tf.keras
kl = ks.layers
cfg = tf.config.experimental
```

For any effective and generalizable allocation strategy, we need to be able to reason about our resources uniformly.

## Normalized virtual GPUs

We start with the new TF functionality of partitioning our physical GPUs into custom-sized, and thus easily "normalizable", virtual GPUs.

The components, or layers of our models, can then expect the ids of the properly sized, or allocated, virtual GPUs.

Given the parameter-driven "resource" requirements of our layers, we can also develop heuristics for partitioning and allocating the physical devices before starting a training session. Such heuristics are beyond the scope of this blog.

```python
devs = ((None, ), (1000, 1000, 1000, 1000, 1000, 1000), (1000, 1000, 1000, 1000, 1000, 1000))
cfg.set_visible_devices(cfg.get_visible_devices('CPU')[:1], 'CPU')
cfg.set_visible_devices(cfg.get_visible_devices('GPU')[:len(devs) - 1], 'GPU')
for d, ms in zip(cfg.get_visible_devices(), devs):
    vs = [cfg.VirtualDeviceConfiguration(m) for m in ms]
    cfg.set_virtual_device_configuration(d, vs)
devs = cfg.list_logical_devices('CPU')
devs += cfg.list_logical_devices('GPU')
print('devices:', [d.name for d in devs])
```

  ```sh
    devices: ['/job:localhost/replica:0/task:0/device:CPU:0', '/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1', '/job:localhost/replica:0/task:0/device:GPU:2', '/job:localhost/replica:0/task:0/device:GPU:3', '/job:localhost/replica:0/task:0/device:GPU:4', '/job:localhost/replica:0/task:0/device:GPU:5', '/job:localhost/replica:0/task:0/device:GPU:6', '/job:localhost/replica:0/task:0/device:GPU:7', '/job:localhost/replica:0/task:0/device:GPU:8', '/job:localhost/replica:0/task:0/device:GPU:9', '/job:localhost/replica:0/task:0/device:GPU:10', '/job:localhost/replica:0/task:0/device:GPU:11']
  ```

Let's turn off "soft" allocation for now:

```python
tf.config.set_soft_device_placement(False)
```

## Pipeline of small GPUs

The model we develop here builds on a configurable "stack" of identical layers.

A most basic, custom `dense` layer class is all we need as the stack's repeated component.

We aim to "lay" this stack on its side, and onto our virtual GPUs, as a functional, forward-backward propagating pipeline that could now fit in our combined GPU-space.

Each layer of the stack would, therefore, use a predetermined, or heuristically pre-calculated, virtual GPU `idx`:

```python
class Layer(kl.Layer):
    def __init__(self, i, ps, **kw):
        super().__init__(**kw)
        self.idx = min(i + 1, len(devs) - 1)
        self.ps = ps

    def build(self, input_shape):
        s = input_shape[-1]
        with tf.device(devs[self.idx].name):
            self.w = self.add_weight(name='l_w', shape=(s, s))
            self.b = self.add_weight(name='l_b', shape=(s, ))
        return super().build(input_shape)

    def call(self, x):
        with tf.device(devs[self.idx].name):
            y = tf.einsum('bi,ij->bj', x, self.w) + self.b
        return y
```

## A sample sequential model

A basic `sequential` Keras model will suffice as the container of our stack.

Once our input, as well as the output, is shaped, we simply chain our chosen number of layers together, in the middle of our basic `sequential` model.

The Keras model's `summary` method is handy to confirm our model is laid out just as intended:

```python
def model_for(ps):
    m = ks.Sequential()
    m.add(kl.Dense(ps.dim_hidden, input_dim=ps.dim_input, name='in'))
    for i in range(ps.num_layers):
        m.add(Layer(i, ps, name=f'lay_{i}'))
    m.add(kl.Dense(ps.dim_input, name='out'))
    m.compile(optimizer=ps.optimizer(), loss=ps.loss(), metrics=[ps.metrics()])
    print(m.summary())
    return m
```

## Symbolic parameters

Before we can run our model, we need to establish our parameters, the various dimensions and other attributes we want to use to shape the training.

A simple Python `dict` works best to keep things organized, unique and also sorted:

```python
params = dict(
    dim_hidden=1000,
    dim_input=100,
    loss=ks.losses.MeanAbsoluteError,
    metrics=ks.metrics.MeanAbsoluteError,
    num_layers=10,
    optimizer=ks.optimizers.SGD,
)
```

The drawback of string keyed `dict`s is just that, the strings can have typos in them and hundreds of potentially misnamed parameters, later on, will certainly cause unnecessary confusion.

Python's automatically verified native `attribute`s come to the rescue once again.

Here is a simple, straightforward and functional `Params` class:

```python
class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
```

Let's create our `Params` instance and a truly handy training data set (with testing and verification all built-in) in just one line of code:

```python
ps = Params(**params)
import numpy as np
d = np.ones((100, ps.dim_input))
```

## 10 million weights to validate with

Finally, we are ready to compile our model. And, just as expected, the `summary` of the model shows that it has over 10 million weights.

The initial values of the weights are randomly picked. Through training, we bring these arbitrary values "inline" through millions of multiplications and additions executed by our many virtual GPUs, only to verify that our input `ones` are, in fact, just a series of `1`s:

```python
m = model_for(ps)
```

  ```sh
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    in (Dense)                   (None, 1000)              101000
    _________________________________________________________________
    lay_0 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_1 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_2 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_3 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_4 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_5 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_6 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_7 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_8 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    lay_9 (Layer)                (None, 1000)              1001000
    _________________________________________________________________
    out (Dense)                  (None, 100)               100100
    =================================================================
    Total params: 10,211,100
    Trainable params: 10,211,100
    Non-trainable params: 0
    _________________________________________________________________
    None
  ```

Training the model gives us the familiar Keras output, showing a nice convergence of a trivial problem across easily configurable GPUs:

```python
from datetime import datetime
ld = datetime.now().strftime('%Y%m%d-%H%M%S')
ld = f'/tmp/logs/{ld}'
cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
m.fit(d, d, callbacks=cs, epochs=10, batch_size=10)
```

  ```sh
    Train on 100 samples
    Epoch 1/10
    100/100 [==============================] - 2s 17ms/sample - loss: 0.4796 - mean_absolute_error: 0.4796
    Epoch 2/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.2872 - mean_absolute_error: 0.2872
    Epoch 3/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.2414 - mean_absolute_error: 0.2414
    Epoch 4/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.2252 - mean_absolute_error: 0.2252
    Epoch 5/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.1988 - mean_absolute_error: 0.1988
    Epoch 6/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.1984 - mean_absolute_error: 0.1984
    Epoch 7/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.1734 - mean_absolute_error: 0.1734
    Epoch 8/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.1551 - mean_absolute_error: 0.1551
    Epoch 9/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.1719 - mean_absolute_error: 0.1719
    Epoch 10/10
    100/100 [==============================] - 1s 7ms/sample - loss: 0.1527 - mean_absolute_error: 0.1527
  ```

And now let's fire up TensorBoard and visually confirm that our stack of "dense" layers is connected just as expected.

If you haven't run the code, an already generated graph is [here](generated/images/tech/gpus.pdf).

```python
#%load_ext tensorboard
#%tensorboard --logdir /tmp/q/logs
```

This concludes our blog, please see how to use the new dataset functionality by clicking on the next blog.
