---
layout: modeling
title: Learn, Adapt And Protect From Predators
description: 
toc: true
---

- TODO: expand bullets

- Deceive, Deceive: The more "elite", the more deceitful
- Low-Cost communication: professional profiling becomes "two way"
- The era of evolutionary competition of mighty AIs

- more detail

Once again, we have already too much accumulated code to load into our notebook. For your refence, we sample from the attached local source files: *datasets.py, layers.py, main.py, modules.py, samples.py, tasks.py, trafo.py* and *utils.py*.

```python
import tensorflow as tf

ks = tf.keras

def train_eager(ps, ds, m):
    def step(x, t):
        with tf.GradientTape() as tape:
            y = m(x)
            loss = ps.loss(t, y)
            loss += sum(m.losses)
            xent = ps.metric(t, y)
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

```python
def evaluate(ps, ds, m):
    mp = pth.Path('/tmp/q/model')
    if tf.train.get_checkpoint_state(str(mp)):
        m.train_on_batch(ds)
        m.load_weights(str(mp / f'{m.name}'))
        loss, xent = m.evaluate(ds)
        print(f'\nEvaluate loss, xent: {loss}, {xent}')
```

```python
def predict(ps, ds, m):
    mp = pth.Path('/tmp/q/model')
    if tf.train.get_checkpoint_state(str(mp)):
        m.train_on_batch(ds)
        m.load_weights(str(mp / f'{m.name}'))
        for x, t in ds:
            y = m.predict(x)
            print(y, t.numpy())
```

This concludes our final blog in this section.
