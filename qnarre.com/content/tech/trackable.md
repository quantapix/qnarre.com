---
layout: tech
title: Trackable - A Pervasive Persistence Infrastructure
description: 
toc: true
---

Training in TensorFlow means continually adjusting collections of values stored as `tensors` in objects called `variables`.

The persistence of these variables, from one training session to the next, is critical for improving on the already achieved, but otherwise long-running results.

A new system-wide pervasive `trackable` architecture now provides just such a persistence infrastructure. Instead of the old name-based hierarchy, the new design applies a topological, "layered objects" naming scheme.

In this first blog, we explore some of the key aspects of this architecture. We start with a high-level view and then we gradually build from the simple base classes to the more useful Keras `layers`.

Our objective is to arrive at a training model representable by the [graph](generated/images/tech/trackable.pdf) and [runnable example](https://github.com/quantapix/qnarre/blob/master/docs/advanced_tf/trackable.ipynb).

We first need to prep our environment to run any meaningful code:

```python
import tensorflow as tf
from datetime import datetime
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import tracking
```

The actual persistence of training data, the `weights`, is ultimately realized through explicit `Checkpoint` objects.

As the number of such representations grows, saved as efficiently encoded versioned files, `CheckpointManager`s help with keeping track (see TF docs for full functionality).

## Private Trackable object

We present a simple scenario for persisting (saving and restoring) a single-valued variable encapsulated by a `Trackable` object as follows:

```python
def trackable(tr1, v):
    c = tf.train.Checkpoint(tr1=tr1)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    p = m.latest_checkpoint
    c.restore(p).expect_partial()
    if p:
        print(f'restored from: {p}')
        print(f'others are: {m.checkpoints}')
    else:
        print('start from scratch')
    print(f'value before: {v.numpy()}')
    v.assign_add(1)
    m.save()
```

Using the above function, our 3 iterations of incrementing the single-valued `int` variable and keeping track of the `Checkpoint` files result in:

```python
tr1 = base.Trackable()
v = tf.Variable(1)
tr1._track_trackable(v, name='tr1_v')
for _ in range(3):
    trackable(tr1, v)
```

  ```sh
    start from scratch
    value before: 1
    restored from: /tmp/q/trackable/ckpt-1
    others are: ['/tmp/q/trackable/ckpt-1']
    value before: 2
    restored from: /tmp/q/trackable/ckpt-2
    others are: ['/tmp/q/trackable/ckpt-1', '/tmp/q/trackable/ckpt-2']
    value before: 3
  ```

While the above snippet is fully functional, the extensive boiler-plate code becomes an unnecessary hassle when implementing even slightly more complex schemes.

Also note that we used a private, undocumented and non-API method to make our code work. A more convenient "auto-tracking" functionality is needed.

## Pythonic AutoTrackable

The native Python attribute mechanism conveniently provides a framework to satisfy such needs as we'll see in a moment.

In preparation, our slightly adjusted printing function is now as follows:

```python
def autotrackable(tr2, tracked, untracked):
    c = tf.train.Checkpoint(tr2=tr2)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    p = m.latest_checkpoint
    c.restore(p).expect_partial()
    if p:
        print(f'restored from: {p}')
    print(f'values before: {tracked.numpy()}, {untracked.numpy()}')
    tracked.assign_add(1000)
    m.save()
    print(f'value as saved: {tracked.numpy()}')
```

And here is our use of an `AutoTrackable` object holding onto 2 single-valued variables.

Notice the intuitive `tr2.v = tracked` assignment, as this is where the entire "trackable" scheme is triggered.

Just in case we want to avoid the default functionality, we can turn off auto-tracking as well:

```python
tr2 = tracking.AutoTrackable()
tracked, untracked = tf.Variable(1000), tf.Variable(0)
tr2.v = tracked
with base.no_automatic_dependency_tracking_scope(tr2):
    tr2.untracked = untracked
for _ in range(2):
    autotrackable(tr2, tracked, untracked)
```

  ```sh
    restored from: /tmp/q/trackable/ckpt-3
    values before: 1000, 0
    value as saved: 2000
    restored from: /tmp/q/trackable/ckpt-4
    values before: 2000, 0
    value as saved: 3000
  ```

Employing the native Python attribute mechanism and assignment operator allows us to reliably "auto track" hundreds or thousands of training variables.

## Layered objects taxonomy

Moreover, a consistent hierarchical "layered objects" naming scheme emerges, without the need for explicit, string-based names.

For a snapshot view of the "topology" of our layers, or just a simple inventory of our variables, we can use the helper functions provided by TF:

```python
def listing():
    c = tf.train.Checkpoint()
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    p = m.latest_checkpoint
    vs = tf.train.list_variables(p)
    print(f'names and shapes list: {vs}')
    n, _ = vs[-1]
    v = tf.train.load_variable(p, n)
    print(f'loaded value: {v} for name: {n}')
    c = tf.train.load_checkpoint(p)
    ts = c.get_variable_to_dtype_map()
    ss = c.get_variable_to_shape_map()
    print(f'checkpoint types: {ts} and shapes: {ss}')
```

Looking at the result of calling our function, we can quickly grasp the otherwise simple pattern of hierarchical naming convention employed by the architecture:

```python
listing()
```

  ```sh
    names and shapes list: [('_CHECKPOINTABLE_OBJECT_GRAPH', []), ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr2/v/.ATTRIBUTES/VARIABLE_VALUE', [])]
    loaded value: 3000 for name: tr2/v/.ATTRIBUTES/VARIABLE_VALUE
    checkpoint types: {'tr2/v/.ATTRIBUTES/VARIABLE_VALUE': tf.int32, '_CHECKPOINTABLE_OBJECT_GRAPH': tf.string, 'save_counter/.ATTRIBUTES/VARIABLE_VALUE': tf.int64} and shapes: {'tr2/v/.ATTRIBUTES/VARIABLE_VALUE': [], '_CHECKPOINTABLE_OBJECT_GRAPH': [], 'save_counter/.ATTRIBUTES/VARIABLE_VALUE': []}
  ```

## Full Python attribute mechanism

Any type of variable management system that allows creating variables must also support deleting them.

The familiar native Python attribute mechanism's `del` operation is used to delete a variable from the hierarchy, just as shown below:

```python
def deleting(tr2):
    c = tf.train.Checkpoint(tr2=tr2)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    c.restore(m.latest_checkpoint)
    c.tr2.deleted = tf.Variable(-1)
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'list deleted: {vs}')
    del c.tr2.deleted
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'deleted IS DELETED: {vs}')
```

And here are the results of calling our `deleting` function:

```python
deleting(tr2)
```

  ```sh
    list deleted: [('_CHECKPOINTABLE_OBJECT_GRAPH', []), ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr2/deleted/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr2/v/.ATTRIBUTES/VARIABLE_VALUE', [])]
    deleted IS DELETED: [('_CHECKPOINTABLE_OBJECT_GRAPH', []), ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr2/v/.ATTRIBUTES/VARIABLE_VALUE', [])]
  ```

## Aggregating variables

Variable management also means possibly aggregating variables into various containers.

Intuitive Python `list` and `dict` structures can be transparently employed through the `trackable` mechanism.

Using our below-modified function to print our variables in our `containers':

```python
def containers(tr3):
    c = tf.train.Checkpoint(tr3=tr3)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'containers: {vs}')
```

Just as mentioned above, we can intuitively collect variables into either `list`s or `dict`s.

And the patterns used for naming the thus aggregated variables are just as expected:

```python
tr3 = tracking.AutoTrackable()
br1 = tracking.AutoTrackable()
br1.v = tf.Variable(5)
br2 = tracking.AutoTrackable()
br2.v = tf.Variable(5)
tr3.br_list = [br1, br2]
br3 = tracking.AutoTrackable()
br3.v = tf.Variable(5)
tr3.br_dict = {'br3': br3}
containers(tr3)
```

  ```sh
    containers: [('_CHECKPOINTABLE_OBJECT_GRAPH', []), ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr3/br_dict/br3/v/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr3/br_list/0/v/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr3/br_list/1/v/.ATTRIBUTES/VARIABLE_VALUE', [])]
  ```

## Sharing variables

Neural networks rely on sharing persisted trainable weights, TF variables in our case, to express interdependencies.

Variable sharing was ad-hoc, only name-based and with a global scope before.

As Python has extensive native support for managing easily sharable references to its objects, this fundamental problem gets an intuitive solution with the new trackable architecture.

As expected, sharing variables now is natural and also safe, as it uses references instead of error-prone strings:

```python
def sharing(tr3):
    c = tf.train.Checkpoint(tr3=tr3)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    c.restore(m.latest_checkpoint).assert_consumed()
    v1 = tr3.br_list[0].v
    v2 = tr3.br_list[1].v
    vd1 = tr3.br_dict['br1'].v
    vd2 = tr3.br_dict['br2'].v
    vd3 = tr3.br_dict['br3'].v
    print(f'all fives: {v1.numpy()}, {v2.numpy()}, {vd3.numpy()}')
    print(f'shared too: {vd1.numpy()}, {vd2.numpy()}')
    v1.assign_add(5)
    v2.assign_add(5)
    vd3.assign_add(5)
    m.save()
    vs = tf.train.list_variables(m.latest_checkpoint)
    print(f'shared not repeated: {vs}')
    v1.assign_add(-10)
    v2.assign_add(-10)
    vd3.assign_add(-10)
    print(f'all zeros: {v1.numpy()}, {v2.numpy()}, {vd3.numpy()}')
    print(f'shared too: {vd1.numpy()}, {vd2.numpy()}')
    c2 = tf.train.Checkpoint(tr3=tr3)
    m = tf.train.CheckpointManager(c2, '/tmp/q/trackable', max_to_keep=2)
    c2.restore(m.latest_checkpoint).assert_consumed()
    print(f'all tens: {v1.numpy()}, {v2.numpy()}, {vd3.numpy()}')
    print(f'shared too: {vd1.numpy()}, {vd2.numpy()}')
```

Persisted shared variables are not duplicated in checkpoints. And when checkpoints are restored or reloaded, the in-memory sharing of variables is also re-established.

Updates to our shared variables can be easily verified just as follows:

```python
tr3.br_dict = {'br1': br1, 'br2': br2, 'br3': br3}
sharing(tr3)
```

  ```sh
    all fives: 5, 5, 5
    shared too: 5, 5
    shared not repeated: [('_CHECKPOINTABLE_OBJECT_GRAPH', []), ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr3/br_dict/br3/v/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr3/br_list/0/v/.ATTRIBUTES/VARIABLE_VALUE', []), ('tr3/br_list/1/v/.ATTRIBUTES/VARIABLE_VALUE', [])]
    all zeros: 0, 0, 0
    shared too: 0, 0
    all tens: 10, 10, 10
    shared too: 10, 10
  ```

## Encapsulating variables

Variable management also means possible encapsulation.

The new `Module` objects build on `AutoTrackable` to extend Python's familiar `class`-based encapsulation mechanism.

The also supported explicit name scoping of modules allows the reuse of module classes, as instances of the same class would need to be generically counted otherwise:

```python
class Module(tf.Module):
    sub = None

    def __init__(self, name=None):
        super().__init__(name=name)
        with self.name_scope:
            self.v = tf.Variable(1, name='m_v')

    def __str__(self):
        s = f'n: {self.name}, v: {self.v.numpy()}'
        if self.sub:
            s += f', s: ({self.sub})'
        return s

    @tf.Module.with_name_scope
    def __call__(self):
        if self.sub is None:
            y = tf.constant(100)
        else:
            y = self.sub()
        y = tf.math.add(y, self.v)
        self.v.assign(y)
        return y
```

When building hierarchies of modules, TF provided convenience methods also allow for recursively collecting the "layered" variables. This is essential for computing gradients:

```python
def modules(mod):
    vs = [v.name for v in mod.variables]
    ms = [m.name for m in mod.submodules]
    print(f'mod variables: {vs}, submodules: {ms}')
    c = tf.train.Checkpoint(module=mod)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    mod()
    print(mod)
    m.save()
    mod()
    print(mod)
    p = m.latest_checkpoint
    vs = tf.train.list_variables(p)
    print(f'containers: {vs}')
    c.restore(p)
    print(f'restored: {mod}')
```

With our module class and our handy printing function, we can now build a basic nested hierarchy of 3 layered modules.

Printing our "one branch tree" shows both the name-based hierarchy and the Python-object or topological "checkpoint" hierarchy:

```python
mod1 = Module('m1')
mod1.sub = Module('m2')
mod1.sub.sub = Module('m3')
modules(mod1)
```

  ```sh
    mod variables: ['m1/m_v:0', 'm2/m_v:0', 'm3/m_v:0'], submodules: ['m2', 'm3']
    n: m1, v: 103, s: (n: m2, v: 102, s: (n: m3, v: 101))
    n: m1, v: 406, s: (n: m2, v: 303, s: (n: m3, v: 201))
    containers: [('_CHECKPOINTABLE_OBJECT_GRAPH', []), ('module/sub/sub/v/.ATTRIBUTES/VARIABLE_VALUE', []), ('module/sub/v/.ATTRIBUTES/VARIABLE_VALUE', []), ('module/v/.ATTRIBUTES/VARIABLE_VALUE', []), ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', [])]
    restored: n: m1, v: 103, s: (n: m2, v: 102, s: (n: m3, v: 101))
  ```

Keras is the API for consistently reasoning about the interconnected network of components. It also visibly splits the two distinct, building vs. executing, phases of our component "graphs".

The `functional` Keras, as opposed to either the `sequential` or the `subclassed` flavors, has the most pre-packaged features to assist us with our neural networks. We aim to use it throughout our blogs.

## Modules and Keras layers

Keras `layer`s, as well-defined encapsulating components, build on the previously used `module`s to manage variable persistence.

Hence, the previous `module`s example is almost identical to the below shown "Keras layers" version:

```python
class Layer(tf.keras.layers.Layer):
    def __init__(self, sub=None, **kw):
        super().__init__(**kw)
        self.sub = sub

    def __str__(self):
        s = f'n: {self.name}, v: {self.v.numpy()}'
        if self.sub:
            s += f', s: ({self.sub})'
        return s

    def build(self, input_shape):
        self.v = self.add_weight(name='l_v',
                                 shape=[],
                                 dtype=tf.int32,
                                 initializer=tf.ones_initializer)
        return super().build(input_shape)

    def call(self, x):
        if self.sub is None:
            y = x
        else:
            y = self.sub(x)
        y = tf.math.add(y, self.v)
        self.v.assign(tf.reduce_sum(y))
        return y
```

And, after adjusting our helper to print the results:

```python
def models(mod, lay):
    print(mod.summary())
    vs = [v.name for v in mod.variables]
    ts = [t.name for t in mod.trainable_variables]
    ms = [m.name for m in mod.submodules]
    print(f'lay variables: {vs}, trainables: {ts}, submodules: {ms}')
    d = tf.constant([100, 100])
    mod(d)
    print(lay)
    c = tf.train.Checkpoint(model=mod)
    m = tf.train.CheckpointManager(c, '/tmp/q/trackable', max_to_keep=2)
    m.save()
    mod(d)
    print(lay)
    p = m.latest_checkpoint
    vs = tf.train.list_variables(p)
    print(f'containers: {vs}')
    c.restore(p)
    print(f'restored: {lay}')
```

## A most simple model

We finally arrive at the most simple Keras model.

It uses just 3 scalar variables to showcase the underlying already tried and used persistence management:

```python
ins = [tf.keras.Input(shape=(), dtype=tf.int32)]
lay = Layer(name='l1', sub=Layer(name='l2', sub=Layer(name='l3')))
outs = [lay(ins)]
mod2 = tf.keras.Model(name='m2', inputs=ins, outputs=outs)
models(mod2, lay)
```

  ```sh
    Model: "m2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None,)]                 0
    _________________________________________________________________
    l1 (Layer)                   (1, None)                 3
    =================================================================
    Total params: 3
    Trainable params: 3
    Non-trainable params: 0
    _________________________________________________________________
    None
    lay variables: ['l1/l_v:0', 'l1/l2/l_v:0', 'l1/l2/l3/l_v:0'], trainables: ['l1/l_v:0', 'l1/l2/l_v:0', 'l1/l2/l3/l_v:0'], submodules: ['input_1', 'l1', 'l2', 'l3']
    n: l1, v: 206, s: (n: l2, v: 204, s: (n: l3, v: 202))
    n: l1, v: 1424, s: (n: l2, v: 1012, s: (n: l3, v: 604))
    containers: [('_CHECKPOINTABLE_OBJECT_GRAPH', []), ('model/layer_with_weights-0/l_v/.ATTRIBUTES/VARIABLE_VALUE', []), ('model/layer_with_weights-0/sub/l_v/.ATTRIBUTES/VARIABLE_VALUE', []), ('model/layer_with_weights-0/sub/sub/l_v/.ATTRIBUTES/VARIABLE_VALUE', []), ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', [])]
    restored: n: l1, v: 206, s: (n: l2, v: 204, s: (n: l3, v: 202))
  ```

Nevertheless, even the simplest model can be overwhelming when expressed only textually.

TensorBoard is an accompanying tool that can help us in "picturing" the nested component graphs.

As "a picture is worth a thousand words", `summary` data for TB is generated as follows:

```python
def graph(tracer):
    s = datetime.now().strftime('%Y%m%d-%H%M%S')
    d = f'/tmp/q/logs/func/{s}'
    w = tf.summary.create_file_writer(d)
    tf.summary.trace_on(graph=True)  # , profiler=True)
    tracer()
    with w.as_default():
        tf.summary.trace_export(name="trace", step=0, profiler_outdir=d)
```

Please note that our trivially simple Keras model still implements data-driven Python recursion.

## New Pythonic 'autograph'

The new `autograph` functionality allows us to use such intuitive, native expressions instead of the usual, but more cumbersome, TF "graph ops".

Autograph code generation is invoked with the `tf.function` Python decorator. A later blog will highlight the most impressive features of this new approach to defining ops.

```python
@tf.function
def tracer2():
    return mod2(tf.constant([100, 100]))
```

In order to see the TB generated summaries, including the picture of our graph, we need to load the extension:

```python
%load_ext tensorboard
```

Then we generate the TB summaries by calling our `tracer` function:

```python
graph(tracer2)
```

And now we can view the zoom-able and clickable TB graph. If you haven't run the code, an already generated graph is [here](generated/images/tech/trackable.pdf).

```python
#%tensorboard --logdir /tmp/q/logs/func
```

This concludes our blog. For using the new GPU-related functionality, please click on our next blog.
