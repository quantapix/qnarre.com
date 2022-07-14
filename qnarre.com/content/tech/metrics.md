---
layout: tech
title: Modular And Reusable Metrics All The Way
description: 
toc: true
---

As we have run various training sessions, and we have approached modeling itself from various angles, it is time for us to consider the piece of TF functionality that ties it all together.

Ultimately, the end goal of any modeling, and of the actual training with our models, is to arrive to simply quantifiable "measures", the `losses`, that we can then use in our repeated iterations of gradient calculations and subsequent tuning the weights of the model.

A `loss` is defined as some sort of difference, or "distance", between the results of our model's calculations and the given targets.

Specifically, we have been using `crossentropy`, (see a fun explanation [here](https://colah.github.io/posts/2015-09-Visual-Information/)) as that "distance".

Just as before, we need to prep our environment to run any meaningful code:

```python
import tensorflow as tf
import dataset as qd
import custom as qc
import autograph as qa
ks = tf.keras
kl = ks.layers
```

In order to be able to experiment with multiple `loss` and `metrics` settings, we duplicate our `tgt` tensors in our newly defined `adapter` function of our `dataset`.

```python
@tf.function
def adapter(d):
    enc, dec, tgt = d['enc'], d['dec'], d['tgt']
    return ((
        enc.flat_values,
        enc.row_splits,
        dec.flat_values,
        dec.row_splits,
        tgt.flat_values,
        tgt.row_splits,
    ), (
        tgt.to_tensor(),
        tgt.to_tensor(),
    ))
```

We also adjust our `ToRagged` layer.

Instead of leaving the `tf.function` decorator generic, which is allowing multiple version of the op to be generated based on the actual shapes of the input tensors, we restrict the generated op to only one version: the one taking triple input tensor pairs of any 1D shape.

```python
class ToRagged(qc.ToRagged):
    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int64)
    ] * 3])
    def call(self, x):
        ys = []
        for i in range(3):
            i *= 2
            fv, rs = x[i:i + 2]
            ys.append(tf.RaggedTensor.from_row_splits(fv, rs))
        return ys
```

## Loss class

And now we are ready to tackle replacement `loss` and `metric` classes.

The Keras `losses.Loss` base class, that all the various other "losses" are derived from, has a `call` method with the above mentioned two arguments: the target tensor and the model's output tensor.

Our replacement implementation of the method skips the various checks and validations from the canned version and simply flattens the two known tensors followed by calling directly the efficient graph op implementation of crossentropy:

```python
class Loss(ks.losses.Loss):
    @staticmethod
    def xent(tgt, out):
        tgt = tf.reshape(tf.cast(tgt, tf.int64), [-1])
        s = tf.shape(out)
        out = tf.reshape(out, [-1, s[-1]])
        y = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt,
                                                           logits=out)
        return tf.reshape(y, s[:-1])

    def __init__(self):
        super().__init__(name='loss')

    def call(self, tgt, out):
        return self.xent(tgt, out)
```

## Metric class

Our `Metric` class is even simpler, it adds the aggregating `total` and `count` variables and then delegates to calling our `xent` function (the same that our matching `Loss` uses).

```python
class Metric(ks.metrics.Metric):
    def __init__(self):
        super().__init__(name='metric', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, tgt, out, sample_weight=None):
        vs = Loss.xent(tgt, out)
        self.total.assign_add(tf.math.reduce_sum(vs))
        return self.count.assign_add(tf.cast(tf.size(vs), dtype=tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)
```

## Updates to our model

Our model needs to be updated to use the newly defined components, including our new `Loss` and `Metric` classes.

As we include both the `Debed` and the `Probe` layers from our previous blogs, and they show up as a pair of output tensors respectively identifiable by their names, we can assign different losses and metrics to each.

We chose to use the same `loss` and `metric` for both. Keras, as expected, will sum both losses and metrics to calculate the end result.

To keep things simple, and since we have a pair of outputs, we also had to double the targets that the `loss` and `metric` would use:

```python
def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = ToRagged()(x)
    y = qc.Frames(ps)(y)
    embed = qc.Embed(ps)
    ye = qc.Encode(ps)(embed(y[:2]))
    yd = qc.Decode(ps)(embed(y[2:]) + [ye[0]])
    y = qc.Debed(ps)(yd)
    ys = qa.Probe(ps)(yd)
    m = ks.Model(inputs=x, outputs=[y, ys])
    m.compile(
        optimizer=ps.optimizer,
        loss={'debed': ps.loss, 'probe': ps.loss},
        metrics={'debed': [ps.metric], 'probe': [ps.metric]},
    )
    print(m.summary())
    return m
```

We also need to update our parameters with instances of our specific `Loss` and `Metric` instances we want Keras to call on:

```python
params = qc.params
params.update(
    loss=Loss(),
    metric=Metric(),
)
```

## Training session

And now we are ready to start our training session.

We can confirm the model's layers and connections and we can easily adjust the parameters to tailor the length of the sessions to our objectives.

```python
ps = qd.Params(**params)
ps.num_epochs = 1
import masking as qm
qm.main_graph(ps, qc.dset_for(ps, adapter).take(10), model_for(ps))
```

  ```sh
    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_1 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_3 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_4 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_5 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_6 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    to_ragged (ToRagged)            [(None, None), (None 0           input_1[0][0]
                                                                     input_2[0][0]
                                                                     input_3[0][0]
                                                                     input_4[0][0]
                                                                     input_5[0][0]
                                                                     input_6[0][0]
    __________________________________________________________________________________________________
    frames (Frames)                 [(5, 25), (None,), ( 125         to_ragged[0][0]
                                                                     to_ragged[0][1]
                                                                     to_ragged[0][2]
    __________________________________________________________________________________________________
    embed (Embed)                   multiple             120         frames[0][0]
                                                                     frames[0][1]
                                                                     frames[0][2]
                                                                     frames[0][3]
    __________________________________________________________________________________________________
    encode (Encode)                 [(None, 25, 6), (Non 90516       embed[0][0]
                                                                     embed[0][1]
    __________________________________________________________________________________________________
    decode (Decode)                 [(None, 15, 6), (Non 54732       embed[1][0]
                                                                     embed[1][1]
                                                                     encode[0][0]
    __________________________________________________________________________________________________
    debed (Debed)                   (None, None, None)   140         decode[0][0]
                                                                     decode[0][1]
    __________________________________________________________________________________________________
    probe (Probe)                   (None, None, None)   140         decode[0][0]
                                                                     decode[0][1]
    ==================================================================================================
    Total params: 145,773
    Trainable params: 145,648
    Non-trainable params: 125
    __________________________________________________________________________________________________
    None
    10/10 [==============================] - 11s 1s/step - loss: 5.8626 - debed_loss: 2.9345 - probe_loss: 2.9280 - probe_metric: 2.9239
  ```

With our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries that TB can conveniently visualize.

If you haven't run the code below, an already generated graph is [here](generated/images/tech/metrics.pdf) and [runnable example](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/metrics.ipynb).

```python
#%load_ext tensorboard
#%tensorboard --logdir /tmp/q/logs
```

This concludes our blog, please see how to use Keras callbacks by clicking on the next blog.
