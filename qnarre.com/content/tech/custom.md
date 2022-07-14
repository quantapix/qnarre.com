---
layout: tech
title: Custom Keras Layers Without The Drawbacks
description: 
toc: true
---

In this blog we continue to train our computer to "understand" elementary symbolic arithmetic.

We slightly change our approach however. Instead of having fixed input/context for our encoder/decoder stacks, we follow the idea of "sliding contexts" from this [paper](https://arxiv.org/pdf/1901.02860.pdf).

In addition, we continue to architect our model with a mixture of "validated" Keras `layers` as well as light-weight `Modules` containing bare TF ops.

Our objective is to ultimately arrive at a model representable by the [graph](generated/images/tech/custom.pdf) and [runnable example](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/custom.ipynb).

Just as before, we need to prep our environment to run any meaningful code:

```python
import tensorflow as tf
import dataset as qd
import layers as ql
ks = tf.keras
kl = ks.layers
```

Before we start, we need to increase our dataset slightly as the training steps are becoming more meaningful.

Calling the `dump_dset` function with a parameters instance will update our stored sharded binary files:

```python
def dump_dset(ps):
    ps.max_val = 10000
    ps.num_samples = 1000  # 100000
    ps.num_shards = 10
    fs = [f for f in qd.dump(ps)]
    ps.dim_batch = 100
    for i, _ in enumerate(qd.load(ps, fs).map(adapter)):
        pass
    print(f'dumped {i} batches of {ps.dim_batch} samples each')
    return fs
```

For verification purposes, loading our already created meta data from the sources gives us:

```python
print(qd.vocab)
print(qd.SPC, qd.SEP, qd.STP)
```

  ```sh
    (' ', ':', '|', 'x', 'y', '=', ',', '+', '-', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    0 1 2
  ```

## "Formatter" expanded

We also need to expand our previously used `formatter`.

As we intend to concatenate subsequent inputs in our "sliding context", we need to end the result feature, `res`, of our samples with our `STP = "|"` token.

We have started to use `tf.debugging.assert`s to increase our confidence in the correctness of our data. Later we will be able to switch these out with the familiar Python `asserts`.

Our `formatter` comes with an other significant adjustment.

We intend to feed both our `encoder` and our `decoder` with inputs. Namely, the encoder gets the concatenated `defs` and `op` features, while the decoder gets either a fully or a partially blanked `res`.

Our dataset will supply an `enc`, a `dec` and a `tgt` (the full correct result of the math expression in the sample) tensors. The `rand_blank` function does the quick (inline) random blanking, or masking, of the arithmetic result to be fed into our `decoder` as the `des` tensor.

```python
@tf.function
def formatter(d):
    ds = tf.RaggedTensor.from_sparse(d['defs'])
    n = ds.nrows()
    os = tf.RaggedTensor.from_sparse(d['op'])
    tf.debugging.assert_equal(n, os.nrows())
    ss = tf.fill([n, 1], qd.SEP)
    enc = tf.concat([ds, ss, os, ss], axis=1)
    rs = tf.RaggedTensor.from_sparse(d['res'])
    tf.debugging.assert_equal(n, rs.nrows())
    tgt = tf.concat([rs, tf.fill([n, 1], qd.STP)], axis=1)

    def rand_blank(x):
        y = x.flat_values
        mv = tf.shape(y)[0]
        s = mv // 2
        i = tf.random.uniform([s], maxval=mv, dtype=tf.int32)[:, None]
        y = tf.tensor_scatter_nd_update(y, i, tf.zeros([s], dtype=tf.int32))
        return x.with_flat_values(y)

    return {'enc': enc, 'dec': rand_blank(tgt), 'tgt': tgt}
```

In order for our dataset to be usable, we also need to update our `adapter`.

We continue to split our input ragged tensors into their components, and as we now have 3 ragged inputs: `enc`, `dec` and `tgt`, the total number of dense input tensors to our model will be 6.

The adapter needs to also supply our `tgt` dense tensor to the canned `loss` and `metrics` components that drive the gradient calculations.

In addition, we chose to add `tgt`, or its two components, to our inputs as well. This duplication gives us the chance of feeding correct arithmetic results into our "sliding context".

```python
@tf.function
def adapter(d):
    enc, dec, tgt = d['enc'], d['dec'], d['tgt']
    return (
        (
            enc.flat_values,
            enc.row_splits,
            dec.flat_values,
            dec.row_splits,
            tgt.flat_values,
            tgt.row_splits,
        ),
        tgt.to_tensor(),
    )
```

## Dataset adapter

Our new dataset creator function, `dset_for` is as follows.

We have added an optionally overridable `adapter` argument to be used later.

```python
def dset_for(ps, adapter=adapter):
    ds = tf.data.TFRecordDataset(list(qd.files(ps)))
    ds = ds.take(100).batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(qd.caster)
    return ds.map(formatter).map(adapter)
```

As we now have 3 pairs of input tensors, that we need to convert back into `RaggedTensor`s, we quickly add a `ToRagged` convenience layer that can be seamlessly eliminated once the Keras `Input`s start properly supporting the `ragged=True` keyword argument:

```python
class ToRagged(kl.Layer):
    @tf.function
    def call(self, x):
        ys = []
        for i in range(3):
            i *= 2
            fv, rs = x[i:i + 2]
            ys.append(tf.RaggedTensor.from_row_splits(fv, rs))
        return ys
```

## Frames layer

The `Frames` layer is the new significant addition to our code.

With every new encoder input sequence of tokens `xe`, it first concatenates the `prev` stored context with `xe` and then stores the result in `ye`.

Then it updates the `prev` variable with the concatenation of `ye` and the passed in correct arithmetic result `xt`. The resulting `prev` is to be used in the next cycle.

The computations are slightly more complex due to using the raggedness of the inputs to satisfy the continuous, seamlessly "sliding context" requirement.

The layer also returns the "row_lengths" tensors for both `enc` and `dec` inputs. They will be used later for propagating the input token sequences' "raggedness".

The entire `Frames` layer works exclusively with tokens, as we don't want to keep stale embedding calculations around in our "sliding context".

```python
class Frames(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps, dtype=tf.int32)  # , dynamic=True)
        s = (ps.dim_batch, ps.width_enc)
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.prev = self.add_weight('prev', shape=s, **kw)

    @tf.function
    def call(self, x):
        xe, xd, xt = x
        ye = tf.concat([self.prev, xe], axis=1)
        el = tf.cast(xe.row_lengths(), dtype=tf.int32)
        ye = tf.gather_nd(ye, self.calc_idxs(el))
        c = self.ps.width_dec - xd.bounding_shape(axis=1, out_type=tf.int32)
        yd = tf.pad(xd.to_tensor(), [[0, 0], [0, c]])
        dl = tf.cast(xd.row_lengths(), dtype=tf.int32)
        p = tf.concat([ye, xt], axis=1)
        tl = tf.cast(xt.row_lengths(), dtype=tf.int32)
        p = tf.gather_nd(p, self.calc_idxs(tl))
        self.prev.assign(p)
        return [ye, el, yd, dl]

    def calc_idxs(self, lens):
        b, w = self.ps.dim_batch, self.ps.width_enc
        y = tf.broadcast_to(tf.range(b)[:, None], [b, w])
        i = tf.range(w)[None, ] + lens[:, None]
        y = tf.stack([y, i], axis=2)
        return y
```

As our `Frames` layer returns fixed-width dense tensors once again, we can re-adjust our carried-over `Embed` layer to use the straight `embedding_lookup` instead.

```python
class Embed(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        s = (ps.dim_vocab, ps.dim_hidden)
        self.emb = self.add_weight('emb', shape=s)

    @tf.function
    def call(self, x):
        y, lens = x
        y = tf.nn.embedding_lookup(self.emb, y)
        y *= y.shape[-1]**0.5
        return [y, lens]
```

## Encode, decode and debed layers

We update the `Encode` and `Decode` layers with the addition of the `tf.function` decorators for the `call` methods.

```python
class Encode(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_enc
        self.encs = [Encoder(self, f'enc_{i}') for i in range(ps.dim_stacks)]

    @tf.function
    def call(self, x):
        y = x
        for e in self.encs:
            y = e(y)
        return y

class Decode(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.width = ps.width_dec
        self.decs = [Decoder(self, f'dec_{i}') for i in range(ps.dim_stacks)]

    @tf.function
    def call(self, x):
        y, ye = x[:-1], x[-1]
        for d in self.decs:
            y = d(y + [ye])
        return y
```

Our `Debed` layer is also largely a carry-over, with the adjustment for the now fixed-width tensors.

```python
class Debed(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.dbd = Dense(self, 'dbd', [ps.dim_hidden, ps.dim_vocab])

    @tf.function
    def call(self, x):
        y, lens = x
        s = tf.shape(y)
        y = tf.reshape(y, [s[0] * s[1], -1])
        y = self.dbd(y)
        y = tf.reshape(y, [s[0], s[1], -1])
        y = y[:, :tf.math.reduce_max(lens), :]
        return y
```

## Updated lightweight modules

We update the `Encoder` and `Decoder` modules with the addition of the `tf.function` decorators for the `__call__` methods:

```python
class Encoder(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, 'refl')
            self.conclude = Conclusion(layer, 'conc')

    @tf.function
    def __call__(self, x):
        y = x
        y = self.reflect(y + [y[0]])
        y = self.conclude(y)
        return y

class Decoder(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        with self.name_scope:
            self.reflect = Attention(layer, 'refl')
            self.consider = Attention(layer, 'cnsd')
            self.conclude = Conclusion(layer, 'conc')

    @tf.function
    def __call__(self, x):
        y, ye = x[:-1], x[-1]
        y = self.reflect(y + [y[0]])
        y = self.consider(y + [ye])
        y = self.conclude(y)
        return y
```

The same applies to our new `Attention` module:

```python
class Attention(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        h = layer.ps.dim_hidden
        self.scale = 1 / (h**0.5)
        with self.name_scope:
            self.q = layer.add_weight('q', shape=(h, h))
            self.k = layer.add_weight('k', shape=(h, h))
            self.v = layer.add_weight('v', shape=(h, h))

    @tf.function
    def __call__(self, x):
        x, lens, ctx = x
        off = tf.math.reduce_max(lens)
        q = tf.einsum('bni,ij->bnj', x[:, -off:, :], self.q)
        k = tf.einsum('bni,ij->bnj', ctx, self.k)
        y = tf.einsum('bni,bmi->bnm', q, k)
        # use lens
        y = tf.nn.softmax(y * self.scale)
        v = tf.einsum('bni,ij->bnj', ctx, self.v)
        y = tf.einsum('bnm,bmi->bni', y, v)
        y = tf.concat([x[:, :-off, :], y], axis=1)
        return [y, lens]
```

The same applies to our new `Conclusion` module as well:

```python
class Conclusion(tf.Module):
    def __init__(self, layer, name):
        super().__init__(name=name)
        self.layer = layer
        ps = layer.ps
        w = layer.width * ps.dim_hidden
        with self.name_scope:
            s = [w, ps.dim_dense]
            self.inflate = Dense(layer, 'infl', s, activation='relu')
            s = [ps.dim_dense, w]
            self.deflate = Dense(layer, 'defl', s, bias=False)

    @tf.function
    def __call__(self, x):
        y, lens = x
        w = self.layer.width
        d = self.layer.ps.dim_hidden
        y = tf.reshape(y, [-1, w * d])
        y = self.inflate(y)
        y = self.deflate(y)
        y = tf.reshape(y, [-1, w, d])
        return [y, lens]
```

To add the `tf.function` decorator to our `Dense` module, we simply inherit from the previous version:

```python
class Dense(ql.Dense):
    @tf.function
    def __call__(self, x):
        return super().__call__(x)
```

## Updates for our model

Our model instance needs to be updated as well to use the newly defined components.

Another significant change is the addition of the "row_lengths" tensor (received directly from the ragged tensors) to all the now fixed-width input and output dense tensors.

Once again, we were able to return to using dense tensors for our inputs, despite the "raggedness" of our samples, because we adopted the "sliding context" strategy, thus smoothly concatenating an entire "history" of inputs and correct arithmetic results, into our "working set":

```python
def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = ToRagged()(x)
    y = Frames(ps)(y)
    embed = Embed(ps)
    ye = Encode(ps)(embed(y[:2]))
    yd = Decode(ps)(embed(y[2:]) + [ye[0]])
    y = Debed(ps)(yd)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m
```

Our parameters need to be expanded with the addition of the values for the now fixed widths of both our `encoder` and `decoder` stacks.

```python
params = dict(
    dim_batch=5,
    dim_dense=150,
    dim_hidden=6,
    dim_stacks=2,
    dim_vocab=len(qd.vocab),
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalCrossentropy(from_logits=True),
    num_epochs=5,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
    width_dec=15,
    width_enc=25,
)
```

## Training session

By firing up our training session, we can confirm the model's layers and connections. The listing of a short session follows.

We can easily adjust the parameters to tailor the length of the sessions to our objectives.

```python
ps = qd.Params(**params)
import masking as qm
qm.main_graph(ps, dset_for(ps), model_for(ps))
```

  ```sh
    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_7 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_8 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_9 (InputLayer)            [(None,)]            0
    __________________________________________________________________________________________________
    input_10 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    input_11 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    input_12 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    to_ragged_1 (ToRagged)          [(None, None), (None 0           input_7[0][0]
                                                                     input_8[0][0]
                                                                     input_9[0][0]
                                                                     input_10[0][0]
                                                                     input_11[0][0]
                                                                     input_12[0][0]
    __________________________________________________________________________________________________
    frames_1 (Frames)               [(5, 25), (None,), ( 125         to_ragged_1[0][0]
                                                                     to_ragged_1[0][1]
                                                                     to_ragged_1[0][2]
    __________________________________________________________________________________________________
    embed_1 (Embed)                 multiple             120         frames_1[0][0]
                                                                     frames_1[0][1]
                                                                     frames_1[0][2]
                                                                     frames_1[0][3]
    __________________________________________________________________________________________________
    encode_1 (Encode)               [(None, 25, 6), (Non 90516       embed_1[0][0]
                                                                     embed_1[0][1]
    __________________________________________________________________________________________________
    decode_1 (Decode)               [(None, 15, 6), (Non 54732       embed_1[1][0]
                                                                     embed_1[1][1]
                                                                     encode_1[0][0]
    __________________________________________________________________________________________________
    debed_1 (Debed)                 (None, None, None)   140         decode_1[0][0]
                                                                     decode_1[0][1]
    ==================================================================================================
    Total params: 145,633
    Trainable params: 145,508
    Non-trainable params: 125
    __________________________________________________________________________________________________
    None
    Epoch 1/5
    20/20 [==============================] - 20s 1s/step - loss: 2.6308 - sparse_categorical_crossentropy: 2.6164
    Epoch 2/5
    20/20 [==============================] - 0s 11ms/step - loss: 2.1488 - sparse_categorical_crossentropy: 2.1325
    Epoch 3/5
    20/20 [==============================] - 0s 11ms/step - loss: 1.8967 - sparse_categorical_crossentropy: 1.8844
    Epoch 4/5
    20/20 [==============================] - 0s 13ms/step - loss: 1.7398 - sparse_categorical_crossentropy: 1.7248
    Epoch 5/5
    20/20 [==============================] - 0s 13ms/step - loss: 1.5818 - sparse_categorical_crossentropy: 1.5736
  ```

With our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries that TB can conveniently visualize.

If you haven't run the code below, an already generated graph is [here](generated/images/tech/custom.pdf).

```python
#%load_ext tensorboard
#%tensorboard --logdir /tmp/q/logs
```

We can also switch over to the new `eager` execution mode.

Once again, this is particularly convenient for experimentation, as all ops are immediately executed. And here is a much shortened `eager` session.

```python
# import ragged as qr
# qr.main_eager(ps, dset_for(ps), model_for(ps))
```

This concludes our blog, please see how to use the `autograph` features with our model by clicking on the next blog.
