---
layout: modeling
title: Predatory Profits Of "High-Conflict" Divorces
description: 
toc: true
---

For a justification of this blog, please consider the following chain of conjectures (referenced "lawyers" would be the **divorce lawyer** types):

- our common law is adversarial, the winner simply takes all
- a family, by definition, is not adversarial, as fundamentally both parents love their children
- however, with no adversarial conflict, there is no lawsuit and no profits for lawyers
- thus our common law applied to families must first turn a family into adversaries
- by definition, either unsolved conflicts or perceived lack of resources create adversaries
- moreover, sustainably intractable conflicts guarantee adversaries for life or "high-conflict"
- however, with no money, i.e. no possible profits for lawyers, there simply cannot be "high-conflicts"
- "high-conflict" cases thus are an ambitious, i.e. ruthless, lawyer's "gold mines" and  job-security
- lawyers are in overabundance and competition is fierce, as one only needs to be a malicious actor
- however, with no "high-conflict", there are no trendsetting, "interesting" cases
- with no trendsetting, there is no ~$500 / hour billing rate for ruthless, and narcissist, "top lawyers"

Accepting the above chain of faultless logic, what can a deeply narcissist divorce lawyer do?

- in cases lacking conflicts, he has only one choice: provoke or **flat-out fabricate a conflict by blatantly lying**, specifically for the Family Court's eager consumption
- if he "leaves money on the table", and neglects exploiting lucrative cases he has already hooked-onto, **he will go hungry** with everyone watching!

## Direct contradictions and conflicts

In this blog we focus on *directly* fabricated conflicts, or flat-out, knowingly stated lies to the Family Courts by our lawyers. We are aided by the strict rules of the Court, as all meaningful communication must already be in (or should be convertible to) textual English "inputs".

Our first goal is to train our computer to "catch the knowingly and directly lying lawyer" by systematically finding direct, irrefutable textual contradictions in *all* of a lawyer's communications.

Current state-of-the-art NLP research (see ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)) has shown that the various proposed mechanism for answering generic semantic correctness questions are exceedingly promising. We use them to train our elementary arithmetic model in telling us if a simple mathematical expression is correct or not.

Please note that we have too much accumulated code to load into our notebook. For your reference, we sample from the attached local source files: *datasets.py, layers.py, main.py, modules.py, samples.py, tasks.py, trafo.py* and *utils.py*.

## Synthesized samples

The `Samples` class randomly generates samples from an on-demand allocated pool of values. The size of the pool is set by the `dim_pool` param and using it with large values helps with keeping the probability distributions in check.

Currently, `Samples` can generate a variety of 10 different groups of samples. In this blog we focus on `yes-no` (YNS), `masked` (MSK), `reversed` (REV) and `faulty` (FIX) samples.

A simple sample generating loop can be as follows:

```python
import samples as qs

groups = tuple('yns ynx msk msx cls clx qas rev gen fix'.split())

YNS, YNX, MSK, MSX, CLS, CLX, QAS, REV, GEN, FIX = groups

def sampler(ps):
    ss = qs.Samples(ps)
    for _ in range(ps.num_samples):
        ss, idx = ss.next_idx
        enc, res, *_ = ss.create(idx)
        dec = tgt = f'[{res}]'
        bad = f'[{ss.other(res)}]'
        yn = ss.yns[0, idx]

        d2 = dec if yn else bad
        yns = dict(enc=enc, dec=d2 + '|_', tgt=d2 + f'|{yn}')

        yield {YNS: yns}
```

The generated samples are Python `dict`s with the previously introduced `enc` (encoder), `dec` (decoder) and `tgt` (target) features.

Both `dec` and `tgt` features end the sample with `|` and the yes-no answer is encoded as `1` and `0` (the `_` is the place-holder that the decoder needs to solve).

And now we can generate a few samples:

```python
import utils as qu

ps = dict(
    dim_pool=3,
    max_val=100,
    num_samples=4,
)
ps = qu.Params(**ps)

for d in sampler(ps):
    print(f'{d[YNS]}')
```

  ```sh
    {'enc': 'x=81,y=11;x+y', 'dec': '[10]|_', 'tgt': '[10]|0'}
    {'enc': 'y=-99,x=-58;x+y', 'dec': '[-157]|_', 'tgt': '[-157]|1'}
    {'enc': 'x=13,y=-79;y-x', 'dec': '[-92]|_', 'tgt': '[-92]|1'}
    {'enc': 'y=-33,x=-30;y+x', 'dec': '[-96]|_', 'tgt': '[-96]|0'}
  ```

While we don't show any of the other samples in this blog, the `MSK` features mask the results at random positions with a `?`, the `REV` samples mix up the order of the variables and `FIX` samples randomly introduce an error digit in the results.

## Model definition

The actual model is largely similar to the models already presented in the previous blogs.

Based on what group of samples we are using, we activate some layers in the model while ignoring others.

A significant consideration is that all the 10 groups of samples contribute (if meaningful) to the same weights (or variables).

We chose to do this this based on the results of the [MT-DNN](https://arxiv.org/pdf/1901.11504.pdf) paper. Varying the type and challenge of the samples we effectively cross-train the model.

In order to clearly separate the `loss` and `metric` calculations between the groups, we create a new instance of our model for each group of samples. However, we reuse the same layers.

To accomplish this, we define an `lru_cache` function:

```python
import functools

@functools.lru_cache(maxsize=32)
def layer_for(cls, *pa, **kw):
    return cls(*pa, **kw)
```

And now, our usual `model_for` function looks as follows:

```python
def model_for(ps, group):
    x = inputs
    y = layer_for(ql.ToRagged)(x)
    yt = layer_for(ql.Tokens, ps)(y)
    ym = layer_for(ql.Metas, ps)(y)
    xe, xd = yt[:2] + ym[:1], yt[2:] + ym[1:]
    embed = layer_for(ql.Embed, ps)
    ye = layer_for(ql.Encode, ps)(embed(xe))[0]
    decode = layer_for(ql.Decode, ps)
    if group in (qs.YNS, qs.YNX):
        y = decode(embed(xd) + [ye])
        y = layer_for(ql.Debed, ps)(y)
    elif group in (qs.MSK, qs.MSX):
        y = layer_for(ql.Deduce, ps, embed, decode)(xd + [ye])
    if group in (qs.QAS, qs.FIX):
        y = decode(embed(xd) + [ye])
        y = layer_for(ql.Locate, ps, group)(y)
    m = Model(name='trafo', inputs=x, outputs=[y])
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m
```

## Expanded parameters

As we have expanded the functionality of our layers and modules from the previous blogs, our params have increased in number.

Also, the subsequent blogs in this section will describe the additions to the model's extended functionality.

```python
import tensorflow as tf
import datasets as qd
ks = tf.keras

params = dict(
    activ_concl='gelu',
    dim_attn=4,
    dim_attn_qk=None,
    dim_attn_v=None,
    dim_batch=5,
    dim_concl=150,
    dim_hidden=6,
    dim_hist=5,
    dim_metas=len(qd.metas),
    dim_stacks=2,
    dim_vocab=len(qd.vocab),
    drop_attn=None,
    drop_concl=None,
    drop_hidden=0.1,
    initer_stddev=0.02,
    loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
    metric=ks.metrics.SparseCategoricalCrossentropy(from_logits=True),
    num_epochs=2,
    num_heads=3,
    num_rounds=2,
    num_shards=2,
    optimizer=ks.optimizers.Adam(),
    width_dec=40,
    width_enc=50,
)

params.update(
    loss=qu.Loss(),
    metric=qu.Metric(),
)
```

And this is our new `main` function that loops through all the samples in all of our groups of samples and either trains the model on the samples or performs an evaluation/prediction.

In the follow-on blogs we present the various training/eval/predict functions that our main loop can use:

```python
def main(ps, fn, groups=None, count=None):
    qu.Config.runtime.is_training = True
    groups = groups or qs.groups
    for r in range(ps.num_rounds):
        for g in groups:
            print(f'\nRound {r + 1}, group {g}...\n=======================')
            fn(ps, qd.dset_for(ps, g, count=count), model_for(ps, g))
```

## Generating samples

Before we start a training session, we need to generate some samples.

The code that generates the samples is similar to the following.

The `large` datasets generate 100 shards containing each 10,000 samples for every every sample group out of the current 10.

The total number of samples for the `large` dataset can be easily varied, however, with the pictured settings, it amounts to **10 million samples** that a server with 40 hyper-threads generates in about 3 hours.

```python
ds_small = dict(
    dim_batch=5,
    dim_pool=10,
    max_val=1000,
    num_samples=20,
    num_shards=2,
)

ds_large = dict(
    dim_batch=1000,
    dim_pool=1024 * 1024,
    max_val=100000,
    num_samples=10000,
    num_shards=100,
)

def dump_ds(kind):
    ps = qu.Params(**(ds_small if kind == 'small' else ds_large))
    ss = [s for s in qd.dump(ps, f'/tmp/q/data/{kind}')]
    ds = qd.load(ps, shards=ss).map(qd.adapter)
    for i, _ in enumerate(ds):
        pass
    print(f'dumped {i + 1} batches of {ps.dim_batch} samples each')
```

And here is an actual call to generate our `small` sample set:

```python
!python main.py
```

  ```sh
    dumping /tmp/q/data/small/cls/shard_0000.tfrecords...
    dumping /tmp/q/data/small/msk/shard_0000.tfrecords...
    dumping /tmp/q/data/small/yns/shard_0000.tfrecords...
    dumping /tmp/q/data/small/qas/shard_0000.tfrecords...
    dumping /tmp/q/data/small/clx/shard_0000.tfrecords...
    dumping /tmp/q/data/small/msx/shard_0000.tfrecords...
    dumping /tmp/q/data/small/ynx/shard_0000.tfrecords...
    dumping /tmp/q/data/small/rev/shard_0000.tfrecords...
    dumping /tmp/q/data/small/gen/shard_0000.tfrecords...
    dumping /tmp/q/data/small/yns/shard_0001.tfrecords...
    dumping /tmp/q/data/small/msk/shard_0001.tfrecords...
    dumping /tmp/q/data/small/cls/shard_0001.tfrecords...
    dumping /tmp/q/data/small/fix/shard_0000.tfrecords...
    dumping /tmp/q/data/small/ynx/shard_0001.tfrecords...
    dumping /tmp/q/data/small/clx/shard_0001.tfrecords...
    dumping /tmp/q/data/small/msx/shard_0001.tfrecords...
    dumping /tmp/q/data/small/qas/shard_0001.tfrecords...
    dumping /tmp/q/data/small/gen/shard_0001.tfrecords...
    dumping /tmp/q/data/small/rev/shard_0001.tfrecords...
    dumping /tmp/q/data/small/fix/shard_0001.tfrecords...
    dumped 80 batches of 5 samples each
  ```

## Training session

Now we are ready to run a short training session:

```python
!python trafo.py
```

  ```sh
    Round 1, group yns...
    =======================
    Epoch 1/2
    2/2 [==============================] - 9s 4s/step - loss: 3.2370 - metric: 3.2364
    Epoch 2/2
    2/2 [==============================] - 0s 84ms/step - loss: 3.2212 - metric: 3.2209
    Round 1, group msk...
    =======================
    Epoch 1/2
    2/2 [==============================] - 32s 16s/step - loss: 3.2135 - metric: 3.2134
    Epoch 2/2
    2/2 [==============================] - 0s 119ms/step - loss: 3.2034 - metric: 3.2032
    Round 1, group qas...
    =======================
    Epoch 1/2
    2/2 [==============================] - 7s 4s/step - loss: 3.4434 - metric: 3.4434
    Epoch 2/2
    2/2 [==============================] - 0s 82ms/step - loss: 2.7450 - metric: 2.7450
    Round 2, group yns...
    =======================
    Epoch 1/2
    2/2 [==============================] - 7s 4s/step - loss: 3.2059 - metric: 3.2070
    Epoch 2/2
    2/2 [==============================] - 0s 79ms/step - loss: 3.1923 - metric: 3.1935
    Round 2, group msk...
    =======================
    Epoch 1/2
    2/2 [==============================] - 29s 14s/step - loss: 3.1887 - metric: 3.1887
    Epoch 2/2
    2/2 [==============================] - 0s 130ms/step - loss: 3.1745 - metric: 3.1744
    Round 2, group qas...
    =======================
    Epoch 1/2
    2/2 [==============================] - 10s 5s/step - loss: 1.9412 - metric: 1.9412
    Epoch 2/2
    2/2 [==============================] - 0s 89ms/step - loss: 1.3604 - metric: 1.3604
  ```

And this concludes our blog, please click on the next blog for more detail.
