---
layout: modeling
title: Representative Metrics Through Statistical Modeling
description: 
toc: true
---

- TODO: expand bullets

- Meta-Data: From black & white to full color
- Probabilistic Modeling: "Quantized" credibility of claims
- Learned Derivatives: coherence, fragmentation and turmoil
- Profiling lawyers and predicting trends in Courtroom

- more detail

Once again, we have already too much accumulated code to load into our notebook. For your reference, we sample from the attached local source files: _datasets.py, layers.py, main.py, modules.py, samples.py, tasks.py, trafo.py_ and _utils.py_.

```python
vocab = tuple(' ')
metas = vocab + tuple('XYS OPS RES'.split())
metas += tuple('ABCDEFGHIJ')
separs = tuple(',;[]|')
vocab += separs
vocab += tuple('xy=$+-*')
vocab += tuple('0123456789')
masks = tuple('?_')
vocab += masks

tokens = {c: i for i, c in enumerate(vocab)}
tokens.update((c[0], i) for i, c in enumerate(metas))

SPC = tokens[vocab[0]]
assert SPC == 0
EOS = tokens[separs[-1]]
MSK = tokens[masks[0]]
```

```python
features = tuple('grp enc dec tgt emt dmt out'.split())

GRP, ENC, DEC, TGT, EMT, DMT, OUT = features

def sampler(ps, groups):
    def to_meta(g, x):
        # print('to_metas x', x)
        g = chr(ord('A') + g)
        m, y = '', x.split(';')
        if len(y) > 1:
            m = 'X' * (len(y[0]) + 1)
            y = y[1:]
        y = y[0].split('[')
        y2 = y[0].split('|')
        m += 'O' * len(y2[0])
        if len(y2) > 1:
            m += g * (len(y2[1]) + 1)
        if len(y) > 1:
            y2 = y[1].split('|')
            m += 'R' * (len(y2[0]) + 1)
            if len(y2) > 1:
                m += g * (len(y2[1]) + 1)
        assert len(x) == len(m)
        # print('to_metas m', m)
        return m

    for s in qs.sampler(ps):

        def to_features(g):
            fs = s[g]
            g = qs.groups.index(g)
            e, d, t, o = fs[ENC], fs[DEC], fs[TGT], fs.get(OUT, '')
            d2 = t if '?' in d else d
            return [f'#{g}', e, d, t, to_meta(g, e), to_meta(g, d2), o]

        yield [to_features(g) for g in groups]
```

We have mentioned already that we are largely reusing the model, specifically its layers and modules, from the first section of our blogs.

In this blog we provide more details of the necessary changes in the modules.

```python
import tensorflow as tf
import layers as ql

ks = tf.keras

class Locate(ql.Layer):
    span, spot = None, None

    def __init__(self, ps, group):
        super().__init__(ps)
        h = self.cfg.dim_hidden
        self.width = w = self.cfg.width_dec
        if group is qs.QAS:
            self.span = qm.Dense(self, 'span', [h * w, 2 * w])
        else:
            assert group is qs.FIX
            self.spot = qm.Dense(self, 'spot', [h * w, w])

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'dim_hidden',
            'width_dec',
        )

    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32)
    ]])
    def call(self, x):
        y, _ = x
        s = tf.shape(y)
        y = tf.reshape(y, [s[0], 1, -1])
        if self.span is not None:
            y = self.span(y)
            y = tf.reshape(y, [s[0], 2, -1])
        elif self.spot is not None:
            y = self.spot(y)
        return y
```

```python
import utils as qu

class Attention(tf.Module):
    out = None

    def __init__(self, layer, name):
        super().__init__(name)
        self.layer = layer
        cfg = layer.cfg
        h = cfg.dim_hidden
        k = cfg.dim_attn_qk or cfg.dim_attn or h
        self.scale = 1 / (k**0.5)
        self.num_heads = n = cfg.num_heads or 1
        v = cfg.dim_attn_v
        if not v:
            assert h % n == 0
            v = h // n
        self.drop_rate = cfg.drop_attn or cfg.drop_hidden
        with self.name_scope:
            self.q = Dense(layer, 'q', [h, n * k])
            self.k = Dense(layer, 'k', [h, n * k])
            self.v = Dense(layer, 'v', [h, n * v])
            if n * v != h:
                self.out = Dense(layer, 'out', [n * v, h])

    @tf.function
    def __call__(self, x):
        inp, lens, ctx = x
        off = tf.math.reduce_max(lens)
        x = inp[:, -off:, :]
        q = self.split_heads(self.q(x))
        k = self.split_heads(self.k(ctx))
        v = self.split_heads(self.v(ctx))
        y = tf.einsum('bnxi,bnci->bnxc', q, k)
        y *= self.scale
        # use lens
        y = tf.nn.softmax(y)
        y = tf.einsum('bnxc,bnci->bnxi', y, v)
        y = self.join_heads(y)
        if self.out is not None:
            y = self.out(y)
        y = self.layer.drop(y, self.drop_rate)
        y = self.layer.norm(x + y)
        y = tf.concat([inp[:, :-off, :], y], axis=1)
        return [y, lens]

    def split_heads(self, x):
        s = tf.shape(x)
        y = tf.reshape(x, [s[0], s[1], self.num_heads, -1])
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        return y

    @staticmethod
    def join_heads(x):
        y = tf.transpose(x, perm=[0, 2, 1, 3])
        s = tf.shape(y)
        y = tf.reshape(y, [s[0], s[1], -1])
        return y
```

```python
class Normalization(tf.Module):
    epsilon = 1e-3

    def __init__(self, layer, name, shape):
        super().__init__(name)
        kw = dict(shape=shape, dtype=tf.float32)
        with self.name_scope:
            self.gamma = layer.add_weight('gamma', initializer='ones', **kw)
            self.beta = layer.add_weight('beta', initializer='zeros', **kw)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None])])
    def __call__(self, x):
        mean, variance = tf.nn.moments(x, -1, keepdims=True)
        kw = dict(offset=self.beta,
                  scale=self.gamma,
                  variance_epsilon=self.epsilon)
        y = tf.nn.batch_normalization(x, mean, variance, **kw)
        return y
```

This concludes our blog, please click on the next blog for further details of the additional changes.
