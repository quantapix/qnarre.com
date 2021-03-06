---
layout: modeling
title: From Fabricated Conflicts To Activism
description: 
toc: true
---

We continue to focus on actionable technical steps that can mold our still easily accessible "elementary arithmetic" model into the foundation of our upcoming, purely natural-language-based training efforts.

Further expanding on the last two blogs, we shift our attention to the `classification` (CLS and CLX), `question-answer` (QAS), `compose` (GEN) groups of samples.

The formulated objectives, along with provided solutions, are loosely based on the novel approaches outlined in the trendsetting [BERT](https://arxiv.org/pdf/1810.04805.pdf) and [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) papers.

Once again, we have already too much accumulated code to load into our notebook. For your refence, we sample from the attached local source files: _datasets.py, layers.py, main.py, modules.py, samples.py, tasks.py, trafo.py_ and _utils.py_.

## Samples generation

Following the previous blogs' pattern, CLS and CLX group of samples are generated by steps similar to the below code.

The seeming complication of a two-step "next_idx/create" is caused by our need to first allocate a large enough pool of random values and then to sequentially draw, or "create", from it. This allows us to maintain the requested probability distribution of the samples.

```python
import numpy as np
randint = np.random.randint
import samples as qs

groups = tuple('yns ynx msk msx cls clx qas rev gen fix'.split())

YNS, YNX, MSK, MSX, CLS, CLX, QAS, REV, GEN, FIX = groups

def sampler(ps):
    ss = qs.Samples(ps)
    for _ in range(ps.num_samples):
        ss2 = None
        ss, idx = ss.next_idx
        enc, res, *_ = ss.create(idx)
        dec = tgt = f'[{res}]'

        ss2, i2 = ss.next_idx
        e2, r2, *_ = ss2.create(i2, use_x=res)

        t2 = np.array(list('+*2'))[randint(3)]
        ss2, i2 = ss2.next_idx
        e2, r2, t2, _ = ss2.create(i2, double_y=(t2 == '2'))
        d2 = f'[{r2}]'
        cls = dict(enc=e2, dec=d2 + '|_', tgt=d2 + f'|{t2}')

        t3 = np.array(list('0+-'))[randint(3)]
        ss2, i2 = ss2.next_idx
        e2, r2, _, t3 = ss2.create(i2, use_x=None if t3 == '0' else res)
        d2 = e2 + f'[{r2}]'
        clx = dict(enc=enc + tgt, dec=d2 + '|_', tgt=d2 + f'|{t3}')

        yield {CLS: cls, CLX: clx}
        ss = ss2
```

And now, if we run our "sampler", we get the following samples:

```python
import utils as qu

ps = dict(
    dim_pool=3,
    max_val=100,
    num_samples=4,
)
ps = qu.Params(**ps)

for d in sampler(ps):
    print(f'CLS: {d[CLS]}')
    print(f'CLX: {d[CLX]}')
```

  ```sh
    CLS: {'enc': 'y=-43,x=-92;x*y', 'dec': '[3956]|_', 'tgt': '[3956]|*'}
    CLX: {'enc': 'y=-57,x=-24;x-y[33]', 'dec': 'x=-93,y=92;y*x[-8556]|_', 'tgt': 'x=-93,y=92;y*x[-8556]|0'}
    CLS: {'enc': 'x=-33,y=-10;x-y', 'dec': '[-23]|_', 'tgt': '[-23]|+'}
    CLX: {'enc': 'x=31,y=-55;x*y[-1705]', 'dec': 'y=-48,x=55;y*x[-2640]|_', 'tgt': 'y=-48,x=55;y*x[-2640]|0'}
    CLS: {'enc': 'x=83,y=-74;x*y', 'dec': '[-6142]|_', 'tgt': '[-6142]|*'}
    CLX: {'enc': 'x=-18,y=-33;y+x[-51]', 'dec': 'y=69,x=$;x-y[-120]|_', 'tgt': 'y=69,x=$;x-y[-120]|-'}
    CLS: {'enc': 'x=40,y=40;x-y', 'dec': '[0]|_', 'tgt': '[0]|2'}
    CLX: {'enc': 'y=19,x=99;y+x[118]', 'dec': 'y=-26,x=95;x-y[121]|_', 'tgt': 'y=-26,x=95;x-y[121]|0'}
  ```

## Modeling intent

The challenge of the CLS samples is simple. We need to train our computer to recognize the `op` (e.g. `+`, `-` or `*`) of our arithmetic expression. There is a twist, however. Both `+` and `-` expressions should be recognized as `additive`, i.e. `+`, expressions.

The challenge of the CLX samples gets a bit more difficult. Just like with the previous "extended", `X`, groups, we need to train the computer to use indirect conjectures.

In the case of CLX, we are given 2 expressions, with the second one possibly using the result of the first one. And then, if the second's result is larger or equal to the first one's, the prediction needs to be an `additive`, or `+`, relationship. Otherwise, it should be a `subtractive` relationship.

There are two special cases for both. If there is a "duplication" in the CLS samples, a `2` should result. And if the second expression doesn't depend on the first expression, then a `0` should be returned in the case of CLX.

Generating the QAS and GEN samples is as follows:

```python
def sampler(ps):
    ss = qs.Samples(ps)
    for _ in range(ps.num_samples):
        ss, idx = ss.next_idx
        enc, res, *_ = ss.create(idx)
        dec = tgt = f'[{res}]'

        r1, r3 = f'{ss.other(res)}', f'{ss.other(res)}'
        r2 = f'[{r1}{res}{r3}]'
        b = '{:0>3d}'.format(len(r1) + 1)
        e = '{:0>3d}'.format(len(r2) - len(r3) - 1)
        qas = dict(enc=enc, dec=r2 + '|', tgt='|', out=f'#{b} {e}')

        d2 = '[' + '_' * (len(tgt) + randint(5))
        gen = dict(enc=enc, dec=d2 + '|', tgt=tgt + '|')

        yield {QAS: qas, GEN: gen}
```

And then, calling the generator:

```python
for d in sampler(ps):
    print(f'CLS: {d[QAS]}')
    print(f'CLX: {d[GEN]}')
```

  ```sh
    CLS: {'enc': 'x=-64,y=-51;x-y', 'dec': '[-22-13-73]|', 'tgt': '|', 'out': '#004 007'}
    CLX: {'enc': 'x=-64,y=-51;x-y', 'dec': '[______|', 'tgt': '[-13]|'}
    CLS: {'enc': 'y=-95,x=22;x*y', 'dec': '[73-209072]|', 'tgt': '|', 'out': '#003 008'}
    CLX: {'enc': 'y=-95,x=22;x*y', 'dec': '[________|', 'tgt': '[-2090]|'}
    CLS: {'enc': 'y=-86,x=-97;y-x', 'dec': '[01130]|', 'tgt': '|', 'out': '#002 004'}
    CLX: {'enc': 'y=-86,x=-97;y-x', 'dec': '[____|', 'tgt': '[11]|'}
    CLS: {'enc': 'y=-2,x=-90;y*x', 'dec': '[-29180-60]|', 'tgt': '|', 'out': '#004 007'}
    CLX: {'enc': 'y=-2,x=-90;y*x', 'dec': '[________|', 'tgt': '[180]|'}
  ```

## Question and answer modeling

One can see from the output that the QAS samples "scramble" 3 possible , but only one correct, answer that our computer needs to choose from.

At this point the answer is in the middle, however, the predicted output needs to be the start and the end positions of the correct result in the seemingly confusing looking provided garbled sequence of digits.

We encode the correct result in `tgt` prefixed with a `#`. As can be seen from our inlined `tokenizer` function from _datasets.py_, if we encounter the `#` prefix, we use the subsequent `int` values verbatim.

Otherwise, we "tokenize" by mapping our characters from the `tokens` dictionary.

```python
import tensorflow as tf

@tf.function
def tokenizer(d):
    def to_tokens(x):
        if len(x) > 0 and chr(x[0]) == '#':
            y = ''.join([chr(c) for c in x[1:]])
            y = [int(v) for v in y.split()]
        else:
            y = [tokens[chr(c)] for c in x]
        return tf.constant(y, tf.int32)

    return {
        k: tf.numpy_function(to_tokens, [v], Tout=tf.int32)
        for k, v in d.items()
    }
```

As for the GEN samples, the requested output is simple: compose the correct result! We only provide a randomly variable empty space marked by `_` characters.

## How will training on these samples help us?

Given a very large amount of possibly confusing text, our first task should be to simplify it. Specifically, looping through all the words of the text, we should simply drop anything that we can `compose` ourselves from our already learned context. This is where GEN becomes truly essential.

The next step in reducing complexity is to `classify` our sentences according to a perhaps pre-defined set of `topics`. Obviously, CLS and the more complex CLX will be necessary for that task.

We need to further segment our text base by finding emerging `narratives`, the center-piece of our efforts. The `question-answer` learned "knowledge", QAS, will help with that.

Once again, we need to point out that all these "skills" are learned by using the same weights or variables. This choice was made based on the latest "multi-task" training results.

We defer running the training session to a follow-up blog in order to avoid repetitive content.

This concludes our blog, please click on the next blog for more detail.
