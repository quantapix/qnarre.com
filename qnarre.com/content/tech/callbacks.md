---
layout: tech
title: Keras Callbacks - Extending Their Scope And Usage
description: 
toc: true
---

As a transitioning piece between our to groups of blogs, this blog is still work in progress.

It concerns the various `callbacks` that we can register during training sessions.

As automating our sessions is an important objective of ours, specifically to enable us to fine-tune our training `hyper-parameters`, we will be adding to this blog as the rest of the next group's blogs materialize.

Just as before, we need to prep our environment to run any meaningful code:

```python
from datetime import datetime
import pathlib as pth
import tensorflow as tf
import dataset as qd
import custom as qc
ks = tf.keras
kl = ks.layers
```

## Updates to our model

And now we are ready to define our model.

As this blog focuses on the actual training process, our model can be reused directly from a previous blog:

```python
def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = qc.ToRagged()(x)
    y = qc.Frames(ps)(y)
    embed = qc.Embed(ps)
    ye = qc.Encode(ps)(embed(y[:2]))
    yd = qc.Decode(ps)(embed(y[2:]) + [ye[0]])
    y = qc.Debed(ps)(yd)
    m = ks.Model(name='callbacks', inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m
```

## Using callbacks

Once the model is defined, we adjust our main calling function.

At this point we define our `callbacks` that should be kept "in loop" during our training session.

Initially we still want to include the standard Keras TensorBoard callbacks.

Additionally, we want to roll our own checkpointing. We choose to use the latest `Checkpoint` and `CheckpointManager` classes (see our [blog](tech/trackable) regarding this topic).

For this we define a custom Keras `Callback` class called `CheckpointCB`. As this callback is only used to save or update our current checkpoint file, it only needs to override the `on_epoch_end` callback.

In the override it simply calls the manager's `save` method.  

To be expanded...

```python
def main_graph(ps, ds, m):
    b = pth.Path('/tmp/q')
    b.mkdir(parents=True, exist_ok=True)
    lp = datetime.now().strftime('%Y%m%d-%H%M%S')
    lp = b / f'logs/{lp}'
    c = tf.train.Checkpoint(model=m)
    mp = b / 'model' / f'{m.name}'
    mgr = tf.train.CheckpointManager(c, str(mp), max_to_keep=3)
    # if mgr.latest_checkpoint:
    #     vs = tf.train.list_variables(mgr.latest_checkpoint)
    #     print(f'\n*** checkpoint vars: {vs}')
    c.restore(mgr.latest_checkpoint).expect_partial()

    class CheckpointCB(ks.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mgr.save()

    cbs = [
        CheckpointCB(),
        ks.callbacks.TensorBoard(
            log_dir=str(lp),
            histogram_freq=1,
        ),
    ]
    m.fit(ds, callbacks=cbs, epochs=ps.num_epochs)
```

We may also need to update our parameters as they relate to our "callback objectives".

To be expanded...

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

And now we are ready to start our training session.

We can confirm the model's layers and connections. We can easily adjust the parameters to tailor the length of the sessions to our objectives.

```python
ps = qd.Params(**params)
main_graph(ps, qc.dset_for(ps), model_for(ps))
```

  ```sh
    Model: "callbacks"
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
    ==================================================================================================
    Total params: 145,633
    Trainable params: 145,508
    Non-trainable params: 125
    __________________________________________________________________________________________________
    None
    Epoch 1/5
    20/20 [==============================] - 15s 733ms/step - loss: 1.3225 - sparse_categorical_crossentropy: 1.3311
    Epoch 2/5
    20/20 [==============================] - 0s 6ms/step - loss: 1.2037 - sparse_categorical_crossentropy: 1.2163
    Epoch 3/5
    20/20 [==============================] - 0s 6ms/step - loss: 1.2067 - sparse_categorical_crossentropy: 1.2187
    Epoch 4/5
    20/20 [==============================] - 0s 6ms/step - loss: 1.1113 - sparse_categorical_crossentropy: 1.1200
    Epoch 5/5
    20/20 [==============================] - 0s 6ms/step - loss: 1.0234 - sparse_categorical_crossentropy: 1.0336
  ```

A quick `ls` into our `/tmp/q/model/callbacks` checkpoint directory shows that our manager is in fact updating the checkpoint files and it is keeping only the last three, just as we expect.

To be expanded...
