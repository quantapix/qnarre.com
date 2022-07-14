---
layout: tech
title: Ragged Tensors For Text Processing
description: 
toc: true
---

`RaggedTensor`s are a newly added feature in TF. They intend to cleanly and efficiently solve the previously mentioned "uneven sequence-length" problem.

While easily convertible, they are different from the more general `SparseTensor`s, as they only allow "ragged edges".

From an implementation point of view, `RaggedTensors` are efficiently represented as `composite tensor`s consisting of 1) a packed sequence of values and 2) a list of indices (effectively "row lengths").

These new composite tensors are purpose-fitted to our sample text processing problem.

One significant advantage of their specific design is the fluid nature of their "duality". On one hand, they can be viewed just as a simple vector of values, for efficient graph ops. On the other hand, they can be used as handy "masks", to selectively extract just the right values from dense tensors.

To demonstrate their use, we set our objective to arrive at a model representable by the [graph](generated/images/tech/ragged.pdf) and [runnable example](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/ragged.ipynb).

Just as before, we need to prep our environment to run any meaningful code:

```python
import tensorflow as tf
import dataset as qd
ks = tf.keras
kl = ks.layers
```

Loading our already created meta data from the sources gives us:

```python
print(qd.vocab)
```

  ```sh
    (' ', ':', '|', 'x', 'y', '=', ',', '+', '-', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
  ```

Our previously introduced `adapter` to our various datasets, themselves loadable from our stored samples, is adjusted slightly: we don't immediately convert the created ragged tensors to dense tensors anymore.

## Keras workaround

While Keras has the intention of supporting ragged input tensors (the `ragged=True` optional keyword argument is already available), passing in our `composite tensors` doesn't seem to work just yet. To work around this problem, we split our ragged tensors into its components, pass the components in and then reassemble them, once inside the model.

```python
@tf.function
def adapter(d):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    ss = tf.fill([ds.nrows(), 1], qd.SEP)
    os = tf.RaggedTensor.from_sparse(d['op'])
    x = tf.concat([ds, ss, os], axis=1)
    y = tf.RaggedTensor.from_sparse(d['res'])[:, :1].to_tensor()
    return (x.flat_values, x.row_splits), y
```

With the adapter configured, our dataset streaming function is simple and looks as expected:

```python
def dset_for(ps):
    ds = tf.data.TFRecordDataset(list(qd.files(ps)))
    ds = ds.batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(qd.caster)
    return ds.map(adapter)
```

## Embedding layer

Just as in our previous blog, our elementary math training problem calls for first embedding the passed-in tokens into more spacious hidden dimensions.

Hence our already defined `Embed` class only needs to be adjusted to work with ragged tensor arguments. Specifically, after reassembling our `RaggedTensor` inputs from its passed-in components, we simply apply our trusty `embedding_lookup` to all the flattened or "bunched-up" tokens:

```python
class Embed(kl.Layer):
    def __init__(self, ps):
        super().__init__(dtype=tf.float32)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight(name='emb', shape=s)

    def call(self, x):
        fv, rs = x
        x = tf.RaggedTensor.from_row_splits(fv, rs)
        y = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.emb, x)
        return y
```

## "Ragged" attention mechanism

Our `Reflect` layer will need slightly more changes.

As the `flat_values` of our ragged inputs loose their batch dimensions, the matrix multiplications become simpler (expressed here through the favorite `einsum`, see [here](https://rockt.github.io/2018/04/30/einsum)).

Since the `attention` mechanism's `q`, `k` and `v` components are element-wise tied to their inputs, the new "raggedness" of the inputs doesn't change their scope. This means that we only need to change the content of our `RaggedTensor`s, the "row_lengths" stay the same as implemented by the `with_values` method.

When we switch over to cross-products, for scoring our calculated attention coefficients, we need to use dense tensors once again. But the actual result of the layer is a `RaggedTensor`.

Since its "raggedness" is the same as the layer's input's raggedness, we can conveniently recreate the same shaped `RaggedTensor` from the respective values of the just calculated dense tensor by using the original ragged input's `row_lengths` method. Very convenient indeed:

```python
class Reflect(kl.Layer):
    def build(self, shape):
        s = shape[-1]
        self.scale = 1 / (s**0.5)
        self.q = self.add_weight(name='q', shape=(s, s))
        self.k = self.add_weight(name='k', shape=(s, s))
        self.v = self.add_weight(name='v', shape=(s, s))
        return super().build(shape)

    def call(self, x):
        q = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.q))
        k = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.k))
        v = x.with_values(tf.einsum('ni,ij->nj', x.flat_values, self.v))
        y = tf.einsum('bsi,bzi->bsz', q.to_tensor(), k.to_tensor())
        y = tf.nn.softmax(y * self.scale)
        y = tf.einsum('bsz,bzi->bsi', y, v.to_tensor())
        y = tf.RaggedTensor.from_tensor(y, lengths=x.row_lengths())
        return y
```

We need to implement similar changes in our `Expand` layer.

"Inflating" our hidden dimensions, for the intrinsic learning purposes, requires a fixed width, hence we immediately convert to a larger dense tensor. However, as we are done with our "ragged" token calculations, we can simply pad the input to the layer to our expected `len_max_input`:

```python
class Expand(kl.Layer):
    def __init__(self, ps):
        super().__init__()
        self.ps = ps

    def call(self, x):
        y = x.to_tensor()
        s = tf.shape(y)[-2]
        y = tf.pad(y, [[0, 0], [0, self.ps.len_max_input - s], [0, 0]])
        return y
```

## Training session

We are ready now to define our model.

We have the two inputs, the two components of our input `RaggedTensor`. We also use our new `Embed`, `Reflect` and `Expand` layers adjusted to work with our sudden "raggedness". The rest of the model is simply carried over from the previous blog:

```python
def model_for(ps):
    x = [
        ks.Input(shape=(), dtype='int32'),  # , ragged=True)
        ks.Input(shape=(), dtype='int64'),
    ]
    y = Embed(ps)(x)
    y = Reflect()(y)
    y = Expand(ps)(y)
    y = kl.Reshape((ps.len_max_input * ps.dim_hidden, ))(y)
    y = kl.Dense(ps.dim_dense, activation='relu')(y)
    y = kl.Dense(ps.dim_vocab, name='dbd', activation=None)(y)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m
```

Our parameters are also unchanged:

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
```

By firing up our training session, we can confirm the model's layers and connections. Also, the listing of a short session follows.

We can easily adjust the parameters to tailor the length of the sessions to our objectives. However, at this point the results are still largely meaningless and extending the trainings is not yet warranted.

```python
ps = qd.Params(**params)
import masking as qm
qm.main_graph(ps, dset_for(ps), model_for(ps))
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
    embed (Embed)                   (None, None, 15)     300         input_1[0][0]
                                                                     input_2[0][0]
    __________________________________________________________________________________________________
    reflect (Reflect)               (None, None, 15)     675         embed[0][0]
    __________________________________________________________________________________________________
    expand (Expand)                 (None, None, 15)     0           reflect[0][0]
    __________________________________________________________________________________________________
    reshape (Reshape)               (None, 300)          0           expand[0][0]
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 150)          45150       reshape[0][0]
    __________________________________________________________________________________________________
    dbd (Dense)                     (None, 20)           3020        dense[0][0]
    ==================================================================================================
    Total params: 49,145
    Trainable params: 49,145
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    Epoch 1/10
    1000/1000 [==============================] - 7s 7ms/step - loss: 1.7148 - sparse_categorical_crossentropy: 1.7148
    Epoch 2/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.4729 - sparse_categorical_crossentropy: 1.4729
    Epoch 3/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.3973 - sparse_categorical_crossentropy: 1.3973
    Epoch 4/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.3517 - sparse_categorical_crossentropy: 1.3517
    Epoch 5/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.3136 - sparse_categorical_crossentropy: 1.3136
    Epoch 6/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.2838 - sparse_categorical_crossentropy: 1.2838
    Epoch 7/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.2500 - sparse_categorical_crossentropy: 1.2500
    Epoch 8/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.2134 - sparse_categorical_crossentropy: 1.2134
    Epoch 9/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.1745 - sparse_categorical_crossentropy: 1.1745
    Epoch 10/10
    1000/1000 [==============================] - 3s 3ms/step - loss: 1.1284 - sparse_categorical_crossentropy: 1.1284
  ```

With our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries that TB can conveniently visualize.

If you haven't run the code, an already generated graph is [here](generated/images/tech/ragged.pdf).

```python
%load_ext tensorboard
#%tensorboard --logdir /tmp/q/logs
```

## Eager execution mode

We can also switch over to the new `eager` execution mode. This is particularly convenient for experimentation, as all ops are immediately executed:

```python
def main_eager(ps, ds, m):
    def step(x, y):
        with tf.GradientTape() as tape:
            logits = m(x)
            loss = ps.loss(y, logits)
            loss += sum(m.losses)
            xent = ps.metric(y, logits)
        grads = tape.gradient(loss, m.trainable_variables)
        ps.optimizer.apply_gradients(zip(grads, m.trainable_variables))
        return loss, xent

    @tf.function
    def epoch():
        s, loss, xent = 0, 0.0, 0.0
        for x, y in ds:
            s += 1
            loss, xent = step(x, y)
            if tf.equal(s % 10, 0):
                e = ps.metric.result()
                tf.print('Step:', s, ', loss:', loss, ', xent:', e)
        return loss, xent

    for e in range(ps.num_epochs):
        loss, xent = epoch()
        print(f'Epoch {e} loss:', loss.numpy(), ', xent:', xent.numpy())
```

And here is a much shortened `eager` session:

```python
ps.num_epochs = 1
main_eager(ps, dset_for(ps).take(100), model_for(ps))
```

  ```sh
    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_5 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_6 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    embed_2 (Embed)                 (None, None, 15)     300         input_5[0][0]
                                                                     input_6[0][0]
    __________________________________________________________________________________________________
    reflect_2 (Reflect)             (None, None, 15)     675         embed_2[0][0]
    __________________________________________________________________________________________________
    expand_2 (Expand)               (None, None, 15)     0           reflect_2[0][0]
    __________________________________________________________________________________________________
    reshape_2 (Reshape)             (None, 300)          0           expand_2[0][0]
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 150)          45150       reshape_2[0][0]
    __________________________________________________________________________________________________
    dbd (Dense)                     (None, 20)           3020        dense_2[0][0]
    ==================================================================================================
    Total params: 49,145
    Trainable params: 49,145
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    Step: 10 , loss: 2.32437706 , xent: 1.38269854
    Step: 20 , loss: 4.57577085 , xent: 1.38721442
    Step: 30 , loss: 1.85324097 , xent: 1.39087987
    Step: 40 , loss: 3.42399216 , xent: 1.39260852
    Step: 50 , loss: 2.88146091 , xent: 1.3952688
    Step: 60 , loss: 1.36529291 , xent: 1.39925313
    Step: 70 , loss: 2.36236858 , xent: 1.40269136
    Step: 80 , loss: 1.86278176 , xent: 1.40463638
    Step: 90 , loss: 3.87270284 , xent: 1.40630019
    Step: 100 , loss: 2.69483709 , xent: 1.40788627
    Epoch 0 loss: 2.694837 , xent: 1.4078863
  ```

This concludes our blog, please see how to avoid "layer-proliferation" complexity by clicking on the next blog.
