---
layout: tech
title: Unified Adaptable Masking That Follows
description: 
toc: true
---

A significant difference between image vs. text processing in machine learning is even vs. uneven input sequence length. Padding uneven textual input to a uniform length is an obvious, natural solution.

Indiscriminate padding can, however, pollute our calculations and introduce unwanted biases. Sometimes it is best to cleanly “mask-out” the padded input with carefully chosen, bias minimizing values.

Repeated, explicit and contextual masking calculations become necessary as a result. Historically such code has been cluttering the otherwise clean "flow of data". Keras’ transparent masking mechanism allows for on-demand custom maskings.

Our objective here is to arrive at a model representable by the [graph](generated/images/tech/masking.pdf) and [runnable example](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/masking.ipynb).

Just as before, we need to prep our environment to run any meaningful code:

```python
import tensorflow as tf
from datetime import datetime
import dataset as qd
ks = tf.keras
kl = ks.layers
```

Loading our already created meta data from the sources gives us:

```python
print(qd.vocab)
print(qd.tokens)
```

  ```sh
  (' ', ':', '|', 'x', 'y', '=', ',', '+', '-', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
  {' ': 0, ':': 1, '|': 2, 'x': 3, 'y': 4, '=': 5, ',': 6, '+': 7, '-': 8, '*': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '8': 18, '9': 19}
  ```

## Prepare the datasets

To "adapt" our existing datasets, we recast our parsed streams and start using the new `RaggedTensor`s instead of the default sparse ones.

We also combine existing `feature`s into new ones by inserting separator tokens between the concatenated pieces.

Before handing the prepared streams of data to Keras, we still need to convert them to dense tensors. Most importantly, we pad the tensors to `len_max_input`, with generic zeros, for uniformity.

```python
@tf.function
def caster(d):
    return {k: tf.cast(v, tf.int32) for k, v in d.items()}

SEP = qd.tokens[':']
print(qd.SEP)

@tf.function
def adapter(d, len_max_input):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    ss = tf.fill([ds.nrows(), 1], qd.SEP)
    os = tf.RaggedTensor.from_sparse(d['op'])
    x = tf.concat([ds, ss, os], axis=1).to_tensor()
    x = tf.pad(x, [[0, 0], [0, len_max_input - tf.shape(x)[-1]]])
    y = tf.RaggedTensor.from_sparse(d['res'])[:, :1].to_tensor()
    return x, y
```

  ```sh
  1
  ```

A newly created function will return the paths to our existing file shards.

And now we are ready to create our datasets, custom-adapted to our problem at hand:

```python
def files(ps):
    d = pth.Path('/tmp/q/dataset')
    for i in range(ps.num_shards):
        i = '{:0>4d}'.format(i)
        yield str(d / f'shard_{i}.tfrecords')

def dset_for(ps):
    ds = tf.data.TFRecordDataset(list(qd.files(ps)))
    ds = ds.batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(qd.caster)
    return ds.map(lambda d: adapter(d, tf.constant(ps.len_max_input)))
```

## Keras masking support

Next, we need to tell our custom Keras layers to support masking. Let's do it once for all of them in our own `Layer` base class. We simply inherit from it for all other layers.

Our first layer, the one receiving the to-be-masked input and needing to specifically calculate the versatile `bool` masking tensor, has to override the `compute_mask` method.

We could also transfer the mask calculation to another layer that would do it as an efficient side-effect of its own tasks. In that case we would use the 2 commented out lines:

```python
class Layer(kl.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.supports_masking = True

class Masking(Layer):
    def __init__(self):
        super().__init__()
        # self._compute_output_and_mask_jointly = True

    def compute_mask(self, x, mask=None):
        return tf.not_equal(x, 0)

    def call(self, x):
        # x._keras_mask = self.compute_mask(x)
        return x
```

In order to turn our impossibly "tight" `int32` tokens into something more useful for machine learning, we need to `Embed` them into a much higher dimensional "space".

Our embedding layer, however, is as simple as it gets: it first creates the embedding table and then does the actual lookup using the input tokens.

Once the embedded values are determined, we apply our straightforward `bool` masking cleanly, always resetting the masked-out, high dimensional values to `0` regardless of any "learned" adjustments.

During layer processing, Keras knows that we want to use the transparently hidden mask tensor from our included `mask=None` keyword argument in the `call` method's signature.

For `autograph`'s sake we need to also explicitly check that the optional `mask` argument is `not None`; a simple intuitive `if mask:` would only trigger "trace execution" instead of "graph execution" in our later blogs.

```python
class Embed(Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight(name='emb', shape=s)

    def call(self, x, mask=None):
        y = tf.nn.embedding_lookup(self.emb, x)
        if mask is not None:
            y *= tf.cast(mask, tf.float32)[:, :, None]
        return y
```

## Attention mechanism with masking

Our self-attention layer, fittingly called `Reflect`, does the absolute minimum required steps to implement the "attention" mechanism of the `transformer` architecture. An excellent, creative explanation of how it works is at [here](http://jalammar.github.io/illustrated-transformer/).

The masking tensor is being automatically supplied to the call by Keras. Once again, we only need to state our intention to mask by adding the `mask=None` keyword argument.

The actual masking calculation, based on our previously created `bool` tensor and specific for this layer only, is outright trivial. It simply replaces the to-be-masked values with large negatives:

```python
class Reflect(Layer):
    def build(self, shape):
        s = shape[-1]
        self.scale = 1 / (s**0.5)
        self.q = self.add_weight(name='q', shape=(s, s))
        self.k = self.add_weight(name='k', shape=(s, s))
        self.v = self.add_weight(name='v', shape=(s, s))
        return super().build(shape)

    def call(self, x, mask=None):
        q = tf.einsum('bsi,ij->bsj', x, self.q)
        k = tf.einsum('bsi,ij->bsj', x, self.k)
        y = tf.einsum('bsi,bzi->bsz', q, k) * self.scale
        if mask is not None:
            # tf.print(' *** applying mask')
            m = tf.logical_not(mask)
            m = tf.cast(m, tf.float32)[:, :, None]
            y += m * -1e9
        v = tf.einsum('bsi,ij->bsj', x, self.v)
        y = tf.einsum('bsz,bzi->bsi', tf.nn.softmax(y), v)
        return y
```

We are now ready to create and compile our Keras `functional` model.

As the objective of this blog is to showcase masking, all the other necessary "plumbing" layers are the canned Keras variety ones.

```python
def model_for(ps):
    x = ks.Input(shape=(ps.len_max_input, ), dtype='int32')
    y = Masking()(x)
    y = Embed(ps)(y)
    y = Reflect()(y)
    y = kl.Reshape((ps.len_max_input * ps.dim_hidden, ))(y)
    y = kl.Dense(ps.dim_dense, activation='relu')(y)
    y = kl.Dense(ps.dim_vocab, name='dbd', activation=None)(y)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m
```

The count of our parameters have slightly increased, otherwise they are the same as before. Please see the previous blog for the justification of the `Params` class and the overall scheme.

```python
params = dict(
    dim_batch=2,
    dim_dense=150,
    dim_hidden=15,
    dim_vocab=len(qd.vocab),
    len_max_input=20,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalCrossentropy(from_logits=True),
    num_epochs=10,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
)

class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
```

## Training session

Once we instantiate our parameters and our dataset, and using the already compiled model, we are ready to start a training session conveniently implemented by the Keras `fit` method.

Our aim is to use as much of the versatility, functionality and error checking that Keras provides, so using the model's `fit` method is all we need for now:

```python
def main_graph(ps, ds, m):
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/q/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=ps.num_epochs)

ps = qd.Params(**params)
main_graph(ps, dset_for(ps), model_for(ps))
```

  ```sh
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None, 20)]              0
    _________________________________________________________________
    masking (Masking)            (None, 20)                0
    _________________________________________________________________
    embed (Embed)                (None, 20, 15)            300
    _________________________________________________________________
    reflect (Reflect)            (None, 20, 15)            675
    _________________________________________________________________
    reshape (Reshape)            (None, 300)               0
    _________________________________________________________________
    dense (Dense)                (None, 150)               45150
    _________________________________________________________________
    dbd (Dense)                  (None, 20)                3020
    =================================================================
    Total params: 49,145
    Trainable params: 49,145
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/10
    1000/1000 [==============================] - 4s 4ms/step - loss: 1.7373 - sparse_categorical_crossentropy: 1.7373
    Epoch 2/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 1.4499 - sparse_categorical_crossentropy: 1.4499
    Epoch 3/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 1.3553 - sparse_categorical_crossentropy: 1.3553
    Epoch 4/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 1.2888 - sparse_categorical_crossentropy: 1.2888
    Epoch 5/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 1.2299 - sparse_categorical_crossentropy: 1.2299
    Epoch 6/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 1.1706 - sparse_categorical_crossentropy: 1.1706
    Epoch 7/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 1.1019 - sparse_categorical_crossentropy: 1.1019
    Epoch 8/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 1.0223 - sparse_categorical_crossentropy: 1.0223
    Epoch 9/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 0.9478 - sparse_categorical_crossentropy: 0.9478
    Epoch 10/10
    1000/1000 [==============================] - 2s 2ms/step - loss: 0.8824 - sparse_categorical_crossentropy: 0.8824
  ```

With our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries that TensorBoard can conveniently visualize.

If you haven't run the below code, an already generated graph is [here](generated/images/tech/masking.pdf).

```python
#%load_ext tensorboard
#%tensorboard --logdir /tmp/q/logs
```

This concludes our blog, please see how to use the new `RaggedTensors` instead of masking by clicking on the next blog.
