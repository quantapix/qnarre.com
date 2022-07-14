---
layout: modeling
title: Psychological Non-Profiles Of American Bar "Frequent Fliers"
description: 
toc: true
---

The American "Divorce Corp." is an alleged ~\$50B+ ["industry"](https://www.divorcecorp.com/). As with any current "be stunned, be shocked and pay attention to me" activist movement, it is hard to know what is true and what is for-profit manipulation behind the linked movie.

Nevertheless, the American Bar Association, Section of Family Law, seemingly acknowledges the existence of ["frequent fliers"](https://www.americanbar.org/content/dam/aba/events/family_law/2014/05/section_of_familylawspringcleconference/11_fri_custody.authcheckdam.pdf), as the effectively and reliably "churning" 10%, the **astronomically profitable "high-conflict" 10%**, of their overall yearly "business".

_Quiz: When more than half of American families get divorced and the ABA-licensed/protected lawyers literally and endlessly "milk" 10% of those cases for a profit of ~\$50B / year, how many innocent and endlessly tortured American children does that involve?_

Sadly, the ABA and the lawyers also acknowledge that this profitable business of theirs is **chronically and sadistically child abusive**. Nevertheless, perhaps inspired by the [Fundamental Theorem of Software Engineering](https://en.wikipedia.org/wiki/Fundamental_theorem_of_software_engineering), the "elite" child abusive lawyers quickly learned that "we can solve any problem by introducing an extra level of indirection".

And they started to formally, with the ABA's apparent blessing, blame their victims, the confused, but otherwise clearly capable and well-off-by-necessity parents, with woo-doo science sound-bites: _"the parents in conflict tend to be more preoccupied with their own identity reconstruction"_.

And per the always successful "indirection" strategy of avoiding blame, they formed a mutually beneficial alliance with the:

- "no skin in the game", i.e. [**not a medical doctor**](https://en.wikipedia.org/wiki/Hippocratic_Oath),
- "science sounding dogma", i.e. [**not scientifically repeatable**, reproducible nor falsifiable](https://www.theatlantic.com/science/archive/2018/11/psychologys-replication-crisis-real/576223/) and
- hence [**"absolute judicial immunity"**](https://www.law.com/ctlawtribune/2019/01/14/absolute-immunity-shields-lawyer-sued-over-guardian-ad-litem-role-appeals-court-rules/?slreturn=20190627150712) practitioners.

Namely, the lawyers work seemingly exclusively with Guardian Ad Litems and/or psychologists. Specifically, they work with the psychologists who have a license, with seemingly absolutely no oversight, to arbitrarily and maliciously profile [half of society](https://owlcation.com/social-sciences/What-Judges-Need-to-Know-About-Narcissistic-Personality-Disorder-in-Custody-Cases) with indefensible allegations, i.e. dogmatic scare-tactics and ruthlessly child abusive manipulations.

Supported by the ABA-sponsored "frequent flier" label, the psychologists claim: _"The NPD will bring a plethora of legal actions that barely make sense or are fully nonsense. They will often burn through attorneys. They will have a history of multiple PFA’s against them. **If they are male (and most are)**, they may have multiple custody issues with multiple women."_

Once you control the psychology, you control the person just as Joseph Stalin's associate claimed [**"Show me the man and I’ll find you the crime"**](https://www.cato.org/policy-report/januaryfebruary-2010/criminalization-almost-everything): _"In his foreword to my book, Alan Dershowitz discusses his time litigating cases in the old Soviet Union. He was always taken by the fact that they could prosecute anybody they wanted because some of the statutes were so vague"_.

## Indirect contradictions and conflicts

Finally getting back to our blog, we now focus on **indirectly** "fabricated conflicts". Our second goal is to train our computer to "catch a chain of falsities" by systematically finding indirect, through a second arithmetic expression in this blog, irrefutable textual contradictions in our inputs.

Once again, we have already too much accumulated code to load into our notebook. For your reference, we sample from the attached local source files: _datasets.py, layers.py, main.py, modules.py, samples.py, tasks.py, trafo.py_ and _utils.py_.

Our previous blog has already generated the necessary `small` set of samples.

If you have not run the code yet, please follow that blog to generate samples. You will get an error if you attempt to run this code without available samples.

In this blog we focus on `yes-no-extended` (YNX) and `masked-extended` (MSX) samples. Here is how we can generate a few samples:

```python
import samples as qs

groups = tuple('yns ynx msk msx cls clx qas rev gen fix'.split())

YNS, YNX, MSK, MSX, CLS, CLX, QAS, REV, GEN, FIX = groups

import utils as qu

ps = dict(
    dim_pool=3,
    max_val=100,
    num_samples=2,
)
ps = qu.Params(**ps)

for d in qs.sampler(ps):
    print(f'{d[YNX]}')
    print(f'{d[MSX]}')
```

  ```sh
    {'enc': 'x=-35,y=-64;y-x[-29]', 'dec': 'y=-7,x=$;y-x[22]|_', 'tgt': 'y=-7,x=$;y-x[22]|1'}
    {'enc': 'x=-35,y=-64;y-x[-29]', 'dec': 'y=-7,x??????[2?]|', 'tgt': 'y=-7,x=$;y-x[22]|'}
    {'enc': 'y=40,x=23;x-y[-17]', 'dec': 'y=-61,x=$;x-y[62]|_', 'tgt': 'y=-61,x=$;x-y[44]|0'}
    {'enc': 'y=40,x=23;x-y[-17]', 'dec': '?=?61?x?$?x-?[?4]|', 'tgt': 'y=-61,x=$;x-y[44]|'}
  ```

We only show the YNX and MSX groups of samples. They are similar to our previous YNS and MSK samples in objective.

Their scope is expanded however. The YNX samples introduce "indirection" into the learned, inherent "knowledge" as it uses 2 expressions, the second one building on, or reusing, the first one's result.

Either of the expressions' results can be incorrect. And after training, the computer will have to effectively perform chained deduction to get the `yes-no` answer right.

The MSX samples also two expressions. And instead of `masking` only a homogenous result, these samples randomly mask the entire second expression.

As the masked expression uses the result of the first one, the above mentioned "indirection" is once again in play.

## Model definition and configuration

We have mentioned already that we are largely reusing the model, specifically its layers and modules, from the first section of our blogs.

In this blog we provide more details of the necessary changes in the layers.

We start with a significant change in our `Layer` base-class, as we have started to support the Keras `get_config` method. We are now able to recreate our layers from checkpoints.

In `utils.py` we extend our `Params` class with a new `Config` class that encapsulates the actually persisted param values:

```python
class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def cfg_items(self, *keys):
        for k in keys:
            yield k, getattr(self, k, None)


class Config(Params):
    runtime = Params(is_training=True, print_toks=False)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._items = kw.items()

    def items(self):
        return self._items
```

Using our new `Config`, we add support for the Keras "config" mechanism in `Layer`:

```python
import tensorflow as tf
ks = tf.keras
ki = ks.initializers

class Layer(ks.layers.Layer):
    initer = None

    @staticmethod
    def cfg_items(ps):
        yield from ps.cfg_items('initer_stddev', )

    @classmethod
    def from_config(cls, cfg):
        return cls(qu.Config(**cfg))

    def __init__(self, ps, **kw):
        kw.setdefault('name', qu.to_snake_case(type(self).__name__))
        kw.setdefault('dtype', tf.float32)
        super().__init__(**kw)
        if isinstance(ps, qu.Config):
            self.cfg = ps
        else:
            self.cfg = qu.Config(**dict(self.cfg_items(ps)))
        cfg = self.cfg
        if cfg.initer_stddev:
            self.initer = ki.TruncatedNormal(stddev=cfg.initer_stddev)

    def get_config(self):
        b = super().get_config().items()
        c = self.cfg.items()
        return dict(list(b) + list(c))
```

## Frames revisited

To demonstrate how only necessary params get saved for each layer, we provide a fragment of the already presented `Frames` class below.

The rest of the class, as well as the derived `Tokens` and `Meta` classes are mostly unchanged.

The only significant difference is that we have started to use the attached instance of the `Config` class instead of the previously used bare `Params`:

```python
class Frames(Layer):
    def __init__(self, ps):
        super().__init__(ps, dtype=tf.int32)
        s = [self.cfg.dim_batch, self.cfg.width_enc]
        kw = dict(initializer='zeros', trainable=False, use_resource=True)
        self.prev = self.add_weight('prev', s, **kw)

    def cfg_items(self, ps):
        yield from super().cfg_items(ps)
        yield from ps.cfg_items(
            'dim_batch',
            'dim_hist',
            'width_dec',
            'width_enc',
        )
```

## "Input signature" optimization

Another addition to the layers in this blog is the addition of `input_signatures` to the `call` methods' `tf.function` definitions.

Since we keep a tight control of the shapes of our "transfer" tensors, there is no need for TF to unnecessarily generate Python code.

We provide the partial `Decode` class example to illustrate this addition:

```python
class Decode(Layer):
    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None, None, None]),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None, None])
    ]])
    def call(self, x):
        y, ye = x[:-1], x[-1]
        for dec in self.decs:
            y = dec(y + [ye])
        return y
```

## Deduce layer

We have also added a new layer to specifically help with the MSK and MSX samples.

The new `Deduce` layer provides us with a graph op to conditionally loop through all the masked, by the `?` token, input positions and one-by-one deduce the target tokens.

The loops obviously are driven by the actual number of masked positions in the individual batches of samples. So loop control needs to be executed by the synthesized graph op.

A partial definition of the class follows:

```python
class Deduce(Layer):
    def __init__(self, ps, embed, decode):
        super().__init__(ps)
        self.embed = embed
        self.decode = decode
        s = [self.cfg.dim_hidden, self.cfg.dim_vocab]
        self.inflate = qm.Dense(self, 'inflate', s)

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

    def deduce(self, x):
        y = self.embed(x[:-1])
        y, lens = self.decode(y + x[-1:])
        y = self.inflate(y)
        y = tf.RaggedTensor.from_tensor(y, lens).to_tensor()
        return y

    def update(self, toks, msks, ctx):
        i = tf.cast(msks, tf.int32)
        i = tf.argmax(i, axis=1, output_type=tf.int32)
        n = tf.shape(msks)[0]
        i = tf.stack([tf.range(n), i], axis=1)
        m = tf.zeros_like(msks)
        m = tf.tensor_scatter_nd_update(m, i, tf.ones([n], tf.bool))
        y = tf.boolean_mask(ctx, m)
        y = tf.math.log_softmax(y)
        y = tf.argmax(y, axis=-1, output_type=tf.int32)
        y = tf.tensor_scatter_nd_update(toks, i, y)
        y = tf.where(tf.logical_and(msks, m), y, toks)
        return y
```

The code above depends on references to the `embed` and `decode` layers, as it has to actively a) `deduce` the new token and b) "re-decode" the context in the `update` method.

The loop is initially set to execute for every token in the width of the input context.

Nevertheless, if there are no `qd.MSK`, or "?", tokens and we have reached our `qd.EOS`, or "|", token for every row on the batch, we end the loop.

With these changes in the code, we can adjust the groups of samples we want to train on, to YNX and MSX, and run the previously presented steps. We skip re-running the training sessions until the last "unifying" blog in this section.

This concludes our blog, please click on the next blog for further details of the additional changes.
