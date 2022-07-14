---
layout: tech
title: Dataset - The Tensor "Highway" On-Ramp
description: 
toc: true
---

Machine learning is about lots and lots of data. Organizing the input data is an error-prone, arduous task.

TensorFlow `datasets` were designed to build complex input data pipelines from simple, reusable pieces with the clear objective of normalizing that process.

This blog shows off some of the useful features of this new approach to "feed the beast".

Before we can run any meaningful code, we first need to prep our environment:

```python
import numpy as np
import pathlib as pth
import tensorflow as tf
```

For convenience, and brevity, let's create some aliases as well:

```python
td = tf.data
tt = tf.train
```

## Elementary arithmetic

While computers were made to add numbers together, they quickly run into insurmountable obstacles if these numbers are presented as simple textual sequences of digits.

We aim to finally "teach" our computer to correctly add and multiply, just as we learned in elementary school. And we start with a simple yet fascinating example, inspired by [here](https://arxiv.org/pdf/1812.02825.pdf).

Our input data consists of `num_samples` (perhaps easily millions) of `"x=-12,y=24:y+x:12"`-like strings, or lines, of texts. These visibly consist of `defs`, `op` and `res` fields (separated by `:`).

Our variables are: `x` and `y`, our "operations" are: `=`, `+`, `-` and `*`, and our variables can be assigned values from: `[-max_val, max_val]`.

The rest of the blogs in this group will continue to build on the below presented results:

```python
vocab = (' ', ':', '|')
vocab += ('x', 'y', '=', ',', '+', '-', '*')
vocab += ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
```

We also intend to make our input data pipeline parametric.

However, the obvious simple and intuitive Python `dict` structures, with literal string keys, are error-prone exactly because of unchecked literals.

## Our Params class again

A few lines of code gives as the `Params` class that leverages the native Python attribute mechanism to validate the names of all our params:

```python
params = dict(
    max_val=10,
    num_samples=4,
    num_shards=3,
)

class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
```

## Synthesize lots of "clean" data

Let's now randomly generate our data, fittingly as a Python generator, and based on a given `Params` instance. For this we define:

```python
def py_gen(ps):
    m, n = ps.max_val, ps.num_samples
    # x, y vals in defs
    vals = np.random.randint(low=1 - m, high=m, size=(2, n))
    # (x, y) order if 1 in defs [0] and op [1], respectively
    ords = np.random.randint(2, size=(2, n))
    # index of ['+', '-', '*']
    ops = np.array(['+', '-', '*'])
    ops.reshape((1, 3))
    ops = ops[np.random.randint(3, size=n)]
    for i in range(n):
        x, y = vals[:, i]
        res = f'x={x},y={y}:' if ords[0, i] else f'y={y},x={x}:'
        o = ops[i]
        res += (f'x{o}y:' if ords[1, i] else f'y{o}x:')
        if o == '+':
            res += f'{x + y}'
        elif o == '*':
            res += f'{x * y}'
        else:
            assert o == '-'
            res += (f'{x - y}' if ords[1, i] else f'{y - x}')
        yield res
```

With our generator defined, it takes just a line of code to create millions of correct "exercises" or samples for our training sessions:

```python
ps = Params(**params)
for s in py_gen(ps):
    print(s)
```

  ```sh
    y=-4,x=4:y-x:-8
    y=4,x=-8:y*x:-32
    y=-8,x=7:y-x:-15
    y=-4,x=9:y+x:5
  ```

## Dataset class

The `tf.data.Dataset` class is the main abstraction for our sequence of elements.

Each element of a dataset is one or more Tensors containing the fields, or `features`, of our `sample` "lines" of elementary math exercises.

Using our "in-memory" generator, we can directly create a TF dataset as follows:

```python
def gen_src(ps):
    ds = td.Dataset.from_generator(
        lambda: py_gen(ps),
        tf.string,
        tf.TensorShape([]),
    )
    return ds
```

And here are the first 2 samples of the now tensor-based sequence:

```python
dg = gen_src(ps)
for s in dg.take(2):
    print(s)
```

  ```sh
    tf.Tensor(b'x=-6,y=0:x+y:-6', shape=(), dtype=string)
    tf.Tensor(b'y=1,x=7:y+x:8', shape=(), dtype=string)
  ```

## Data pipelines and ops

An input data pipeline starts with a "source" dataset, perhaps just as simple as the above.

This "source" can also be a `range`, `from_tensor_slices`, `from_tensors` and even a `TextLineDataset` (see the TF docs).

```python
def src_dset(ps):
    ds = np.array(list(py_gen(ps)))
    ds = td.Dataset.from_tensor_slices(ds)
    return ds
```

All datasets can then be consumed one-by-one as iterables or as aggregatables (e.g. using reduce ops) collections.

Datasets also allow chaining of handy "transformations" to themselves. Some of the canned operations are the intuitive: *cache, concatenate, enumerate, reduce, repeat, shuffle, skip, take, zip*.

An example of 2 samples of a new dataset, concatenated with all 4 samples of the previous, gen-based, dataset and also "enumerated" on-the-fly is as follows:

```python
ds = src_dset(ps)
for i, s in ds.take(2).concatenate(dg).enumerate():
    print(i, s)
```

  ```sh
    tf.Tensor(0, shape=(), dtype=int64) tf.Tensor(b'x=9,y=-9:y+x:0', shape=(), dtype=string)
    tf.Tensor(1, shape=(), dtype=int64) tf.Tensor(b'x=-4,y=2:y+x:-2', shape=(), dtype=string)
    tf.Tensor(2, shape=(), dtype=int64) tf.Tensor(b'x=5,y=-7:y-x:-12', shape=(), dtype=string)
    tf.Tensor(3, shape=(), dtype=int64) tf.Tensor(b'y=-4,x=1:x+y:-3', shape=(), dtype=string)
    tf.Tensor(4, shape=(), dtype=int64) tf.Tensor(b'y=2,x=4:x-y:2', shape=(), dtype=string)
    tf.Tensor(5, shape=(), dtype=int64) tf.Tensor(b'y=-6,x=-5:y-x:-1', shape=(), dtype=string)
  ```

We can also filter our to be "pipeline" at any stage, with the objective of perhaps dropping unfit samples:

```python
@tf.function
def filterer(x):
    r = tf.strings.length(x) < 15
    tf.print(tf.strings.format('filtering {}... ', x) + ('in' if r else 'out'))
    return r

for i, s in enumerate(ds.filter(filterer)):
    print(i, s)
```

  ```sh
    filtering "x=9,y=-9:y+x:0"... in
    0 tf.Tensor(b'x=9,y=-9:y+x:0', shape=(), dtype=string)
    filtering "x=-4,y=2:y+x:-2"... out
    filtering "x=-2,y=3:x*y:-6"... out
    filtering "x=-3,y=5:y-x:8"... in
    1 tf.Tensor(b'x=-3,y=5:y-x:8', shape=(), dtype=string)
  ```

## "Filaments" of data

More importantly, we can split the pipeline into named "filaments" of data.

This new feature proves to be extremely useful, allowing us to standardize and unify all our data sources with configurable, on-the-fly channeling of features aggregated therein:

```python
@tf.function
def splitter(x):
    fs = tf.strings.split(x, ':')
    return {'defs': fs[0], 'op': fs[1], 'res': fs[2]}

for s in ds.map(splitter).take(1):
    print(s)
```

  ```sh
    {'defs': <tf.Tensor: id=207, shape=(), dtype=string, numpy=b'x=9,y=-9'>, 'op': <tf.Tensor: id=208, shape=(), dtype=string, numpy=b'y+x'>, 'res': <tf.Tensor: id=209, shape=(), dtype=string, numpy=b'0'>}
  ```

Another example of a "pipeline component" is an in-line Python `dict`-based tokenizer:

```python
tokens = {c: i for i, c in enumerate(vocab)}

@tf.function
def tokenizer(d):
    return {
        k: tf.numpy_function(
            lambda x: tf.constant([tokens[chr(c)] for c in x]),
            [v],
            Tout=tf.int32,
        )
        for k, v in d.items()
    }

for s in ds.map(splitter).map(tokenizer).take(1):
    print(s)
```

  ```sh
    {'defs': <tf.Tensor: id=258, shape=(8,), dtype=int32, numpy=array([ 3,  5, 19,  6,  4,  5,  8, 19], dtype=int32)>, 'op': <tf.Tensor: id=259, shape=(3,), dtype=int32, numpy=array([4, 7, 3], dtype=int32)>, 'res': <tf.Tensor: id=260, shape=(1,), dtype=int32, numpy=array([10], dtype=int32)>}
  ```

## Sharded binary storage

Datasets can be potentially very large, fitting only on disk and in many files.

As transparent data-pipeline performance is key for training throughput, datasets can also be efficiently encoded into binary sequences stored in `sharded` files.

The following will convert our samples into such binary "records":

```python
def records(dset):
    for s in dset:
        fs = tt.Features(
            feature={
                'defs': tt.Feature(int64_list=tt.Int64List(value=s['defs'])),
                'op': tt.Feature(int64_list=tt.Int64List(value=s['op'])),
                'res': tt.Feature(int64_list=tt.Int64List(value=s['res'])),
            })
        yield tt.Example(features=fs).SerializeToString()
```

And we can "dump" our tokenized, ready-to-consume samples into shards of files stored in a directory.

Once these prepared samples are stored, we can "stream" them straight into our models without any more prep (see subsequent blogs):

```python
def shards(ps):
    for _ in range(ps.num_shards):
        yield src_dset(ps).map(splitter).map(tokenizer)

def dump(ps):
    d = pth.Path('/tmp/q/dataset')
    d.mkdir(parents=True, exist_ok=True)
    for i, ds in enumerate(shards(ps)):
        i = '{:0>4d}'.format(i)
        p = str(d / f'shard_{i}.tfrecords')
        print(f'dumping {p}...')
        with tf.io.TFRecordWriter(p) as w:
            for r in records(ds):
                w.write(r)
        yield p

fs = [f for f in dump(ps)]
```

  ```sh
    dumping /tmp/q/dataset/shard_0000.tfrecords...
    dumping /tmp/q/dataset/shard_0001.tfrecords...
    dumping /tmp/q/dataset/shard_0002.tfrecords...
  ```

## Loading data

For streaming, or loading the "records" back, we need to create templates used in interpreting the stored binary data.

With the templates defined, loading them back in, straight into our datasets, can be just as follows.

Note that the names of the shard files are conveniently returned by our "dump" function:

```python
features = {
    'defs': tf.io.VarLenFeature(tf.int64),
    'op': tf.io.VarLenFeature(tf.int64),
    'res': tf.io.VarLenFeature(tf.int64),
}

def load(ps, files):
    ds = td.TFRecordDataset(files)
    if ps.dim_batch:
        ds = ds.batch(ps.dim_batch)
        return ds.map(lambda x: tf.io.parse_example(x, features))
    return ds.map(lambda x: tf.io.parse_single_example(x, features))
```

Before we actually start using the loaded data in our models, let's "adapt" the pipeline to supply dense tensors instead of the originally configured sparse ones.

Also, since we haven't batched anything yet, we set `dim_batch` to `None`:

```python
@tf.function
def adapter(d):
    return [
        tf.sparse.to_dense(d['defs']),
        tf.sparse.to_dense(d['op']),
        tf.sparse.to_dense(d['res']),
    ]

ps.dim_batch = None
for i, s in enumerate(load(ps, fs).map(adapter)):
    print(i, len(s))
```

  ```sh
    0 3
    1 3
    2 3
    3 3
    4 3
    5 3
    6 3
    7 3
    8 3
    9 3
    10 3
    11 3
  ```

The listing above reveals that we merged 3 sharded files, worth 4 samples each, into the 12 printed samples. We only printed the number of features for each sample, for brevity.

**Please also note** how the above in-line adapter converted our named features into unnamed, positional, i.e. in-a-list features. This was necessary as the Keras `Input` doesn't recognize named input tensors yet.

## Batching data

If we turn on batching in our dataset, the same code will now return the following:

```python
ps.dim_batch = 2
for i, s in enumerate(load(ps, fs).map(adapter)):
    print(i, s)
```

  ```sh
    0 (<tf.Tensor: id=1362, shape=(2, 8), dtype=int64, numpy=
    array([[ 4,  5, 14,  6,  3,  5,  8, 11],
           [ 3,  5, 11,  6,  4,  5,  8, 12]])>, <tf.Tensor: id=1363, shape=(2, 3), dtype=int64, numpy=
    array([[4, 7, 3],
           [4, 8, 3]])>, <tf.Tensor: id=1364, shape=(2, 2), dtype=int64, numpy=
    array([[13,  0],
           [ 8, 13]])>)
    1 (<tf.Tensor: id=1368, shape=(2, 9), dtype=int64, numpy=
    array([[ 4,  5,  8, 16,  6,  3,  5,  8, 16],
           [ 4,  5,  8, 12,  6,  3,  5,  8, 14]])>, <tf.Tensor: id=1369, shape=(2, 3), dtype=int64, numpy=
    array([[3, 8, 4],
           [3, 9, 4]])>, <tf.Tensor: id=1370, shape=(2, 1), dtype=int64, numpy=
    array([[10],
           [18]])>)
    2 (<tf.Tensor: id=1374, shape=(2, 8), dtype=int64, numpy=
    array([[ 3,  5,  8, 11,  6,  4,  5, 10],
           [ 3,  5, 12,  6,  4,  5, 14,  0]])>, <tf.Tensor: id=1375, shape=(2, 3), dtype=int64, numpy=
    array([[4, 7, 3],
           [4, 8, 3]])>, <tf.Tensor: id=1376, shape=(2, 2), dtype=int64, numpy=
    array([[ 8, 11],
           [12,  0]])>)
    3 (<tf.Tensor: id=1380, shape=(2, 8), dtype=int64, numpy=
    array([[ 4,  5, 15,  6,  3,  5,  8, 11],
           [ 4,  5, 10,  6,  3,  5, 19,  0]])>, <tf.Tensor: id=1381, shape=(2, 3), dtype=int64, numpy=
    array([[4, 9, 3],
           [3, 7, 4]])>, <tf.Tensor: id=1382, shape=(2, 2), dtype=int64, numpy=
    array([[ 8, 15],
           [19,  0]])>)
    4 (<tf.Tensor: id=1386, shape=(2, 8), dtype=int64, numpy=
    array([[ 4,  5, 12,  6,  3,  5, 10,  0],
           [ 4,  5, 19,  6,  3,  5,  8, 17]])>, <tf.Tensor: id=1387, shape=(2, 3), dtype=int64, numpy=
    array([[4, 8, 3],
           [3, 9, 4]])>, <tf.Tensor: id=1388, shape=(2, 3), dtype=int64, numpy=
    array([[12,  0,  0],
           [ 8, 16, 13]])>)
    5 (<tf.Tensor: id=1392, shape=(2, 8), dtype=int64, numpy=
    array([[ 4,  5,  8, 13,  6,  3,  5, 17],
           [ 4,  5, 10,  6,  3,  5, 14,  0]])>, <tf.Tensor: id=1393, shape=(2, 3), dtype=int64, numpy=
    array([[4, 7, 3],
           [3, 9, 4]])>, <tf.Tensor: id=1394, shape=(2, 1), dtype=int64, numpy=
    array([[14],
           [10]])>)
  ```

As preparation for the subsequent blogs, let's generate a more substantial data source with 10 shards of 1,000 samples each:

```python
ps.max_val = 100
ps.num_samples = 1000
ps.num_shards = 10
fs = [f for f in dump(ps)]
ps.dim_batch = 100
for i, _ in enumerate(load(ps, fs).map(adapter)):
    pass
print(i)
```

  ```sh
    dumping /tmp/q/dataset/shard_0000.tfrecords...
    dumping /tmp/q/dataset/shard_0001.tfrecords...
    dumping /tmp/q/dataset/shard_0002.tfrecords...
    dumping /tmp/q/dataset/shard_0003.tfrecords...
    dumping /tmp/q/dataset/shard_0004.tfrecords...
    dumping /tmp/q/dataset/shard_0005.tfrecords...
    dumping /tmp/q/dataset/shard_0006.tfrecords...
    dumping /tmp/q/dataset/shard_0007.tfrecords...
    dumping /tmp/q/dataset/shard_0008.tfrecords...
    dumping /tmp/q/dataset/shard_0009.tfrecords...
    99
  ```

This concludes our blog, please see how easy masking our uneven sample "lines" can be by clicking on the next blog.
