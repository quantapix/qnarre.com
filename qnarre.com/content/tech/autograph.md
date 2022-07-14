---
layout: tech
title: Autograph - Intuitive Data-Driven Control At Last
description: 
toc: true
---

Continuing our blogs, we shift our focus to yet another new feature in TF, the `autograph` functionality.

Previously, and as far as intuitive expression of code was concerned, "graph ops" efficiently solved complex calculations while failed at simple, sequential control.

By generating on-demand Python code now, `autograph` transparently patches all the necessary graph ops together and packages the result into a "python op".

While the generated new ops are potentially faster than the code before them, in this blog we are more interested in the new expressive powers of the `autograph` package.

Specifically, we look at what becomes possible when decorating our functions with the new `tf.function` decorator, as doing this would by default invoke the `autograph` functionality.

Our objective is to ultimately arrive at a model as represented by the [graph](generated/images/tech/autograph.pdf) and [runnable example](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/autograph.ipynb).

Just as before, we need to prep our environment to run any meaningful code:

```python
import numpy as np
import tensorflow as tf
import dataset as qd
import custom as qc
import layers as ql
ks = tf.keras
kl = ks.layers
```

## Embed op

Next, we borrow the `pos_timing` function from our previous blogs, and override it to return a constant "timing signal" tensor, depending on the `width` and `depth` arguments.

As our first task is to implement a "python branch" in our new `Embed` op, we will be using two different "timing" tensors, one for the `encode` input and the other for the `decode` input.

```python
def pos_timing(width, depth):
    t = ql.pos_timing(width, depth)
    t = tf.constant(t, dtype=tf.float32)
    return t
```

The `Embed` layer will thus create the two constant tensors to be sourced in the subsequent `call` methods.

Our model will call the shared `Embed` instance for both of our stacks. As we have decorated its `call` method with `tf.function`, we can use familiar and intuitive Python comparisons to branch on the value of tensors on-the-fly, during graph execution.

Clearly, our two stacks, while having the same `depth`s, have different `width`s. Also the constant "timing" tensors have different `width`s as well.

Yet we are still able to pick-and-match the otherwise incompatible tensors and successfully add them together, all depending on the actual `width` of our "current" input tensor:

```python
class Embed(qc.Embed):
    def __init__(self, ps):
        super().__init__(ps)
        self.enc_p = pos_timing(ps.width_enc, ps.dim_hidden)
        self.dec_p = pos_timing(ps.width_dec, ps.dim_hidden)

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, lens = x
        y = tf.nn.embedding_lookup(self.emb, y)
        s = tf.shape(y)
        if s[-2] == self.ps.width_enc:
            y += tf.broadcast_to(self.enc_p, s)
        elif s[-2] == self.ps.width_dec:
            y += tf.broadcast_to(self.dec_p, s)
        else:
            pass
        y *= tf.cast(s[-1], tf.float32)**0.5
        return [y, lens]
```

## Frames op

Next we demonstrate how on-the-fly "python ops" can also provide insights into inner processes and data flows.

We borrow our `Frames` layer from the previous blog and override its `call` method with a `tf.function` decorated new version that, besides calling `super().call()`, also calls a new `print_row` Python function on every row in our batch.

Yes, we are calling a Python function and printing its results in a TF graph op while never leaving our intuitive and familiar Python environment! Isn't that great?

The `print_row` function itself is simple, it iterates through the tokens of the "row", it does a lookup of each in our `vocab` "table" for the actual character representing the token and then it "joins" all the characters and prints out the resulting string.

And, if we scroll down to the listing of our training session, we can actually see the "sliding context" of our samples as they fly by during our training.

Needless to say, the listing confirms that our `Frames` layer does a good job concatenating the varied length sample inputs, the target results, as well as the necessary separators.

As a result, a simple Python function, usable during graph ops, provides us invaluable insights deep into our inner processes and data flow.

```python
class Frames(qc.Frames):
    @tf.function
    def call(self, x):
        y = super().call.python_function(x)
        tf.print()

        def print_row(r):
            tf.print(
                tf.numpy_function(
                    lambda ts: ''.join([qd.vocab[t] for t in ts]),
                    [r],
                    Tout=[tf.string],
                ))
            return r

        tf.map_fn(print_row, self.prev)
        return y
```

## Deduce and Probe ops

Our next new layer is the partial `Deduce` layer, showcasing how control is intuitive at last from data-driven branching to searching.

This layer will be used in the next group of blogs as a replacement for our previous `Debed` layer. It contains a tensor-dependent `for` loop to iteratively replace our masked characters with "deduced" ones.

The future `Probe` layer, building on the `Deduce` scheme, implements an approximation of "Beam Search", see [paper](https://arxiv.org/pdf/1702.01806.pdf).

It effectively iterates through the hidden dimensions of the output, and based on parallel `topk` searches, comparing various choices for "debedding" the output, it settles on an "optimal" debedding and thus final token output for our `decoder`.

Without `autograph` the data-driven looping/branching graph ops would have to be expressed in a much more convoluted manner:

```python
"""
class Deduce(Layer):
    @tf.function
    def call(self, x):
        toks, *x = x
        if self.cfg.runtime.print_toks:
            qu.print_toks(toks, qd.vocab)
        y = self.deduce([toks] + x)
        n = tf.shape(y)[1]
        p = tf.shape(toks)[1] - n
        for i in tf.range(n):
            t = toks[:, :n]
            m = tf.equal(t, qd.MSK)
            if tf.equal(tf.reduce_any(m), True):
                t = self.update(t, m, y)
                if self.cfg.runtime.print_toks:
                    qu.print_toks(t, qd.vocab)
                toks = tf.pad(t, [[0, 0], [0, p]])
                y = self.deduce([toks] + x)
            else:
                e = tf.equal(t, qd.EOS)
                e = tf.math.count_nonzero(e, axis=1)
                if tf.equal(tf.reduce_any(tf.not_equal(e, 1)), False):
                    break
        return y
"""
class Probe(ql.Layer):
    def __init__(self, ps):
        super().__init__(ps)
        self.dbd = qc.Dense(self, 'dbd', [ps.dim_hidden, ps.dim_vocab])

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

Our model needs to be updated as well to use the newly defined components.

Other than that, we are ready to start training:

```python
def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = qc.ToRagged()(x)
    y = Frames(ps)(y)
    embed = Embed(ps)
    ye = qc.Encode(ps)(embed(y[:2]))
    yd = qc.Decode(ps)(embed(y[2:]) + [ye[0]])
    y = Probe(ps)(yd)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m
```

## Training session

By firing up our training session, we can confirm the model's layers and connections. The listing of a short session follows.

We can easily adjust the parameters to tailor the length of the sessions to our objectives.

```python
ps = qd.Params(**qc.params)
ps.num_epochs = 1
import masking as qm
qm.main_graph(ps, qc.dset_for(ps).take(10), model_for(ps))
```

  ```sh
    Model: "model_3"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to
    ==================================================================================================
    input_19 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    input_20 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    input_21 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    input_22 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    input_23 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    input_24 (InputLayer)           [(None,)]            0
    __________________________________________________________________________________________________
    to_ragged_3 (ToRagged)          [(None, None), (None 0           input_19[0][0]
                                                                     input_20[0][0]
                                                                     input_21[0][0]
                                                                     input_22[0][0]
                                                                     input_23[0][0]
                                                                     input_24[0][0]
    __________________________________________________________________________________________________
    frames_3 (Frames)               [(5, 25), (None,), ( 125         to_ragged_3[0][0]
                                                                     to_ragged_3[0][1]
                                                                     to_ragged_3[0][2]
    __________________________________________________________________________________________________
    embed_3 (Embed)                 multiple             120         frames_3[0][0]
                                                                     frames_3[0][1]
                                                                     frames_3[0][2]
                                                                     frames_3[0][3]
    __________________________________________________________________________________________________
    encode_3 (Encode)               [(None, 25, 6), (Non 90516       embed_3[0][0]
                                                                     embed_3[0][1]
    __________________________________________________________________________________________________
    decode_3 (Decode)               [(None, 15, 6), (Non 54732       embed_3[1][0]
                                                                     embed_3[1][1]
                                                                     encode_3[0][0]
    __________________________________________________________________________________________________
    probe_3 (Probe)                 (None, None, None)   140         decode_3[0][0]
                                                                     decode_3[0][1]
    ==================================================================================================
    Total params: 145,633
    Trainable params: 145,508
    Non-trainable params: 125
    __________________________________________________________________________________________________
    None
    ["        y=76,x=34:y-x:42|"]
    ["       y=12,x=33:x*y:396|"]
    ["     y=-13,x=-80:x-y:-67|"]
    ["      x=51,y=-70:x-y:121|"]
    ["       y=-24,x=30:x-y:54|"]
          1/Unknown - 3s 3s/step - loss: 3.0014 - sparse_categorical_crossentropy: 3.0014
    ["x:42|x=-15,y=-38:y*x:570|"]
    ["x*y:396|x=36,y=72:y-x:36|"]
    ["y:-67|x=-93,y=55:y+x:-38|"]
    ["-y:121|x=2,y=-66:y-x:-68|"]
    ["x-y:54|x=-1,y=-59:x*y:59|"]
          2/Unknown - 3s 1s/step - loss: 2.9652 - sparse_categorical_crossentropy: 2.9652
    ["570|y=59,x=-78:x*y:-4602|"]
    [":36|y=-98,x=-78:y*x:7644|"]
    [":-38|x=-21,y=-36:y+x:-57|"]
    ["y-x:-68|x=21,y=40:y+x:61|"]
    ["x*y:59|y=31,x=-12:x+y:19|"]
          3/Unknown - 4s 1s/step - loss: 2.9888 - sparse_categorical_crossentropy: 2.9956
    ["y:-4602|y=59,x=66:y-x:-7|"]
    [":7644|y=21,x=67:x*y:1407|"]
    ["x:-57|x=-51,y=-69:x-y:18|"]
    [":61|y=49,x=-70:y*x:-3430|"]
    ["x+y:19|y=53,x=15:x*y:795|"]
          4/Unknown - 5s 1s/step - loss: 2.9879 - sparse_categorical_crossentropy: 2.9924
    [":y-x:-7|y=52,x=50:x-y:-2|"]
    ["1407|y=-86,x=40:y-x:-126|"]
    ["-y:18|x=48,y=-43:y-x:-91|"]
    [":-3430|x=99,y=-24:x+y:75|"]
    ["795|x=94,y=-79:x*y:-7426|"]
          5/Unknown - 6s 1s/step - loss: 2.9691 - sparse_categorical_crossentropy: 2.9697
    ["y:-2|x=17,y=-37:x*y:-629|"]
    ["126|x=99,y=-94:y*x:-9306|"]
    ["x:-91|y=-82,x=63:x+y:-19|"]
    [":75|x=-51,y=-79:x*y:4029|"]
    ["426|y=-67,x=-44:x*y:2948|"]
          6/Unknown - 7s 1s/step - loss: 2.9654 - sparse_categorical_crossentropy: 2.9654
    ["y:-629|y=72,x=28:y+x:100|"]
    ["306|y=93,x=-67:x*y:-6231|"]
    ["-19|y=83,x=-61:y*x:-5063|"]
    ["4029|x=-19,y=-63:x+y:-82|"]
    [":2948|y=-5,x=-31:y*x:155|"]
          7/Unknown - 8s 1s/step - loss: 2.9354 - sparse_categorical_crossentropy: 2.9323
    ["x:100|x=42,y=83:x*y:3486|"]
    [":-6231|x=-8,y=23:x-y:-31|"]
    ["*x:-5063|x=7,y=40:y+x:47|"]
    ["-82|y=-63,x=-35:y*x:2205|"]
    ["155|y=-68,x=-17:y*x:1156|"]
          8/Unknown - 9s 1s/step - loss: 2.9268 - sparse_categorical_crossentropy: 2.9247
    [":3486|x=97,y=30:y*x:2910|"]
    ["y:-31|y=-50,x=-71:y-x:21|"]
    ["y+x:47|x=44,y=59:x+y:103|"]
    ["*x:2205|x=23,y=66:y-x:43|"]
    ["1156|y=-90,x=76:y-x:-166|"]
          9/Unknown - 10s 1s/step - loss: 2.8991 - sparse_categorical_crossentropy: 2.8989
    [":2910|x=-20,y=72:x-y:-92|"]
    ["y-x:21|y=-1,x=91:y-x:-92|"]
    ["+y:103|x=-14,y=0:y+x:-14|"]
    [":43|x=-78,y=64:y*x:-4992|"]
    ["166|x=-81,y=16:y*x:-1296|"]
    10/10 [==============================] - 11s 1s/step - loss: 2.8581 - sparse_categorical_crossentropy: 2.8533
  ```

With our TensorBoard `callback` in place, the model's `fit` method will generate the standard summaries that TB can conveniently visualize.

If you haven't run the code below, an already generated graph is [here](generated/images/tech/autograph.pdf).

```python
#%load_ext tensorboard
#%tensorboard --logdir /tmp/q/logs
```

This concludes our blog, please see how to use customize the losses and metrics driving the training by clicking on the next blog.
