---
layout: post
title: "Understanding the origin of the J-Space"
description: "How the Jacobian lens follows naturally from first-order calculus, the chain rule, and the model's own vocabulary readout."
date: 2026-07-27
math: true
---

If you've read Anthropic's findings on discovering an LLM's "J-Space" and how it represents its "global workspace" and "internal thoughts," they may seem somewhat esoteric. The researchers mention that the key to discovering this space was something called a "Jacobian" and pretty much leave it at that.

Digging through the actual paper, we find that they define the probe they use, the Jacobian lens, in Section 2.1 of [Anthropic’s paper](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens) on the J-Space. However, its central expression seems to just fall out of the blue:

$$
\operatorname{softmax}
\left(
W_U\,\operatorname{norm}(J_\ell h_\ell)
\right)
$$

Why should multiplying an intermediate activation by an averaged Jacobian, and then applying the model’s unembedding matrix, reveal anything meaningful about what the model is representing? Indeed, are there any justified reasons why we should expect this expression to even correspond to something like "internal thoughts"? Or, as some may argue for other findings in ML research, is this expression just the result of luck or "divine benevolence"?

I toyed around with some math for a bit, and I found something remarkable that the original researchers were likely more or less aware of but didn't emphasize enough in the paper. The J-Lens expression does not come out of nowhere—far from it. It follows naturally from the following question that is more or less central to interpretability:

> If we slightly perturb the residual stream at a particular layer, how does that perturbation affect what the model may eventually say?

By using tensors and calculus to answer this question, the structure of J-Lens follows almost immediately. Then, averaging over contexts and positions allows this structure to be refined into a general-purpose reading tool. Ultimately, "discovering" J-Lens is very natural, and in this post, I hope to provide a guide to this discovery process, even if you know fairly little about calculus, tensors, or even how LLMs work.

## The background

An LLM processes text through a sequence of transformer layers. The highway of data that flows from one layer to the next is known as the residual stream, which we will denote by $h$. The residual stream is essentially the model’s evolving internal state through the layers.

Let

$$
h_{(\ell)}^{tm}
$$

denote the residual-stream activation at some layer $\ell$. (For now, think of the superscripts and subscripts as just labels that index the many data points the object contains.) The index $m$ ranges over $d_{\mathrm{model}}$ values that represent a model state. The model dimension $d_{\mathrm{model}}$ is typically high (in the thousands) to adequately represent the wide space of representations we want the model to hold. However, the residual stream does not hold just one representation of dimension $d_{\mathrm{model}}$, but many ordered representations. This ordering is indexed by $t$, known as the position. Therefore, the residual stream can be thought of as a *sequence of representations*, each of which is a list of $d_{\mathrm{model}}$ numbers (a vector), ordered by the $t$ index; this sequence evolves from layer to layer.

By the final layer, the residual stream is now ready to be used for predicting tokens in the model's vocabulary at each position. We will call this final residual stream activation $h_{(f)}^{sn}$, where we clarify that:

- $m$ and $n$ are indices for residual-stream coordinates at an intermediate layer $\ell$ and final layer $f$, respectively.
- $t$ and $s$ index positions at an intermediate layer $\ell$ and final layer $f$, respectively.

We pass the final residual stream through unembedding weights $W$ to obtain a set of logits (unnormalized scores) $z$ over the whole model vocabulary indexed by $v$. $z$ is a new object that gives information about which token the model is likely to output at any position: exactly what we want the model to yield. So the final unembedding process looks like:

$$
h_{(f)}(\mathrm{position},\,\mathrm{coordinate}) \xrightarrow{W} z(\mathrm{position},\,\mathrm{token})
$$

How can we represent this transformation mathematically? We can elucidate this with a completely analogous but perhaps more intuitive example.

Suppose you are making a ticketing program for a multi-day event with a set of known visitors. Each visitor has a ticket with a bunch of information on it: their seat number, VIP status, first-time comer status, etc. (In some sense, these parameters *represent* each unique visitor, wink wink.)

You're tasked with making a program that gives the correct ticket price for each visitor on each day of the event. Fortunately, you already have a set of mathematical formulas that take in all the parameters (seat number, VIP status, etc.) and spit out a single ticket price for each day. Specifically, they are a set of linear formulas that differ for each day of the event. For example, Day 1 is $0.8\,(\mathrm{seat}) + 1.1\,(\mathrm{VIP}) + \cdots$, Day 2 is $1.0\,(\mathrm{seat}) + 1.3\,(\mathrm{VIP}) + \cdots$, etc. We see that all the formulas do is *weight* each parameter and sum them. Therefore, the tool you want should do the following:


$$
\mathrm{tickets}(\mathrm{visitor},\,\mathrm{parameter}) \xrightarrow{\mathrm{tool?}} \mathrm{prices}(\mathrm{visitor},\,\mathrm{day})
$$

How do you do this? Well, it's pretty clear that, for each visitor, you want to multiply each of their parameters by the corresponding weight in the formula. You repeat this process for each day. So maybe you could first make a table of the weights $W$, where the columns and rows label the visitor parameters and the days! This would make a correspondence between each visitor parameter $p$ and the weighting it has for each day $d$.

In practice, what would this look like? Let's say that we want a final table of prices indexed by the visitor and the day they visit. How do we find each entry in the table? To obtain the correct price for one particular visitor on one particular day, we need a sum that looks like:

$$
\mathrm{parameter}_1 \, W(\mathrm{parameter}_1,\,\mathrm{day}) + \mathrm{parameter}_2 \, W(\mathrm{parameter}_2,\,\mathrm{day}) + \cdots
$$

(Aside: This type of expression should be familiar to those with linear algebra experience as an inner product between vectors. Indeed, the list of parameters for a visitor and the list of weights for a day are two vectors, and the final price is an inner product between these vectors.) So, for a particular visitor and particular day, we compose the following sum:

$$
\mathrm{Prices}^{\mathrm{visitor},\,\mathrm{day}} = \sum_{\mathrm{parameters}} \mathrm{Ticket}^{\mathrm{visitor},\,\mathrm{parameter}} W_{\mathrm{parameter}}{}^{\mathrm{day}}
$$

Specifically, we look up the visitor in our table of tickets with parameters. Then, we sum the products of each parameter and the corresponding weight for that parameter and day. "Parameter" is an index we iterate and sum over, whereas "visitor" and "day" are fixed. From this method of finding each entry in the table of prices, we can fill out the final table of prices for *all* visitors and *all* days. Note that the weight table $W$ was the key tool that served as a mapping from the *ticket* information to the *prices* information.

Taking the residual stream to token logits is completely analogous. Instead of a sequence of visitors with unique representations from the parameters they entered, we have a sequence of unique representations from the data flowing through the model: the residual stream. And instead of determining appropriate prices for a multitude of days from visitor representations, we are determining appropriate logits for a multitude of vocabulary tokens from residual stream representations. In both cases, we use a table of weights to map from the representations to the output.

Therefore, we can analogously write:

$$
z^{sv} = \sum_n h_{(f)}^{sn}W_n{}^v.
$$

In other words, we obtain token logits for all positions $s$ and all vocabulary tokens $v$ by summing products of the residual stream $h$ and the unembedding weights $W$ over each coordinate $n$. These unembedding weights establish strengths between directions in the residual-stream space and vocabulary tokens (in a similar way that $W$ for tickets established strengths between visitor parameters and specific days).

We can also write this composition compactly as

$$
z^{sv} = h_{(f)}^{sn}W_n{}^v.
$$

by omitting the sum symbol with the understanding that if an index is repeated both "on top" and "on bottom," it is to be summed over. This is known as "Einstein summation notation" and (at least in a limited sense, without getting into the foundations of tensors) motivates why I've been placing some indices as superscripts and others as subscripts. Note that the summed index disappears in the final composed tensor: $h$ and $W$ are indexed by $s$, $n$, and $v$, but only $s$ and $v$ appear in the final tensor of logits.

This notation makes it clear that the collections of data points we are dealing with are mathematical objects known as *tensors*. This is not necessarily an important point for the explanation that follows. Nevertheless, tensors are foundational objects that I find make explaining what happens in LLMs much clearer and more flexible than explicit vectors and matrices.

## Begin with a perturbation

Now, let's consider the main goal at hand. We are trying to understand how an LLM "thinks" and whether that information can be extracted in some way. It should make sense that a natural place to start is the residual stream itself—it is a high-dimensional representation of the model state as it progresses from earlier layers (immediately after processing input text and its previous output) to verbalizing what it might say next in its later layers.

We can imagine that if the model has "thoughts," they may be extractable from this residual stream, especially in intermediate layers far from both the input and output. Moreover, we can imagine that different directions in the high-dimensional residual stream space should correspond to different kinds of thoughts, or "concepts." For example, there could be a unique direction in the residual stream space that corresponds to "car" and another that corresponds to "mango," and so forth. Note, however, that unlike the ticketing example, these directions aren't labeled like the visitor parameters. Therefore, there is no way to simply look at a particular dimension in the residual stream and immediately know "what that dimension means."

But maybe we can try something else. Suppose a small change is made (in some direction) to the residual stream at layer $\ell$ and position $t$. That change propagates through the remaining layers and alters the final residual stream at positions $s \ge t$. Via the unembedding weights, this induces a change in the token logit values. Therefore, what we want is *a mapping between changes in the residual stream and changes in token output logits*. From what we know already, the flow starting from some intermediate layer $\ell$ is:

$$
\delta h_{(\ell)} \xrightarrow{?} \delta h_{(f)} \xrightarrow{W} \delta z
$$

$\delta$ here just means a small perturbation, and we recall that the unembedding weights $W$ always map from final-layer representations to vocabulary logits. With such a mapping, we establish a correspondence between small steps in "logit space" and small steps in "residual space," even if the desired residual-stream data is deep in the model! This means we now have a tool that admits any desired vocabulary-token direction and returns a corresponding direction in the residual stream for any layer and position we want! But before we get ahead of ourselves, how do we map between the perturbations in the residual stream that reside in different layers?

The idea we want to use is quite simple. If we have a function $f$ of multiple variables $x_1, x_2, \ldots$, then the following approximation holds when $\delta x_a$ is sufficiently small:

$$
\frac{\delta f(x_1, x_2, \ldots)}{\delta x_a} \approx \frac{\partial f(x_1, x_2, \ldots)}{\partial x_a}
$$

In words, the ratio of the variation in $f$ to the variation that causes it in a variable $x_a$ can be approximated by the partial derivative of $f$ with respect to $x_a$. In fact, this approximation becomes the *definition* of the partial derivative in the limit $\delta x_a \rightarrow 0$.

In our case, the function we want to approximate is the final residual stream $h_{(f)}$ from perturbations in an earlier layer residual stream $\delta h_{(\ell)}$. With correct indices, the partial derivative approximation for our case becomes the following (with slight rearrangement):

$$
\delta h_{(f)}^{sn}
\approx
\delta h_{(\ell)}^{tm}
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}.
$$

We clearly see that the missing tool we needed to translate between perturbations is the partial derivative below, which includes four (!) indices: $s$, $n$, $t$, and $m$. They index the coordinates and positions of the residual stream at the final layer and the intermediate, perturbing layer:

$$
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}
$$

This kind of tensor, which iterates over the first derivatives of a function, is known as a *Jacobian*, although it is not yet the final Jacobian tensor we will use to reinvent J-Lens.

Finally, although this is not relevant for the rest of this post, I believe it is important to say that the prior approximation using a partial derivative is known as a *first-order approximation*. A series of terms known as a Taylor series approximates the change in a function caused by a finite, nonzero change in its input variables. The first term makes the largest contribution to this approximation and contains a first-order derivative. Subsequent terms refine the approximation further and involve higher-order derivatives. Without getting into obstructive detail, a better approximation of the final residual-stream perturbation would be written as follows if we included a second-order term:

$$
\delta h_{(f)}^{sn}
\approx
\delta h_{(\ell)}^{tm}
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}
+ \frac{1}{2}\,
\delta h_{(\ell)}^{tm}\,
\delta h_{(\ell)}^{ur}
\frac{\partial^2 h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}\,\partial h_{(\ell)}^{ur}}
+ \mathcal O\!\left(\lVert\delta h_{(\ell)}\rVert^3\right).
$$

J-Lens is inherently a first-order method, so I anticipate that similar second-order interpretability methods will soon be explored as well.

Continuing with our first-order method, recall that the corresponding change in the output logits is

$$
\delta z^{sv} = \delta h_{(f)}^{sn}W_n{}^v.
$$

We can then substitute the first-order propagation equation to obtain

$$
\boxed{
\delta z^{sv}
\approx
\delta h_{(\ell)}^{tm}
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}
W_n{}^v
}
$$

This completes the mapping we originally proposed and is almost the complete J-Lens expression.

It contains, in order:

1. The perturbation at the layer we want to interpret.
2. The Jacobian describing how that perturbation propagates downstream.
3. The unembedding direction describing how the final residual stream affects the logit for vocabulary token $v$.

Therefore, we can interpret the following composition of tensors

$$
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}
W_n{}^v
$$

as a tensor that does something remarkably concrete: it gives the direction, in the coordinate system of layer $\ell$, along which a small movement would increase the later logit for token $v$.

This can be alternatively described as pulling the token’s unembedding direction backward through the model. All this means is that instead of assuming that the word “bug,” for example, has the same vector representation at every layer, we ask the model itself:

> Which direction at this particular layer tends to become downstream evidence for saying “bug”?

## Why average over contexts?

Now, we must exercise caution because the Jacobian above is specific to one particular prompt and to particular source and target positions. It therefore reflects both a general relationship and the accidental details of that context.

For example, a representation of "bug" might be used differently while reading code than while reading about insects. Attention patterns, surrounding concepts, and the model’s current task can all change how the representation is routed downstream.

What we should do next is isolate our mapping from prompts, source positions, and present or future target positions. A clear way to do this is simply to average over them in the Jacobian, represented mathematically as follows ($\mathbb E$ means "expectation value," i.e., "mean"):

$$
\mathbb E_{c,t,s \ge t}
\left[
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}
\right] := J_{(\ell)m}{}^{n}
$$

Here $c$ indexes a particular prompt or context. After averaging, the position indices disappear, leaving one tensor indexed by $n$ and $m$, the residual-stream coordinates at the final and intermediate layers. Since this is still a type of Jacobian tensor, we name this tensor $J$.

This averaging does not make the result completely independent of context. It makes it representative of the distribution of contexts used to construct it. Context-specific effects that vary from example to example tend to weaken, while stable relationships reinforce one another.

Nevertheless, this new tensor $J$ allows us to address a more general question:

> Across many situations, how does a direction at layer $\ell$ tend to affect the final residual stream now or later?

Anthropic describes this as separating a representation’s general disposition to be verbalized from the particular use being made of it in one prompt. Their implementation averages over source positions, later positions, and one thousand pretraining-like prompts.

## From causal sensitivity to a reading tool

We can now compose the averaged Jacobian with the unembedding direction for token $v$:

$$
J_{(\ell)m}{}^n W_n{}^v.
$$

For each vocabulary item $v$, this produces a direction

$$
\widetilde J_{(\ell)m}{}^v
$$

in the residual-stream space of layer $\ell$. It is the average, layer-local direction associated with increasing future evidence for that token. In Anthropic's original language, this tensor holds a model's complete set of "J-vectors" over its vocabulary. Note that the indices make clear that $\widetilde J$ explicitly maps between any token in a model's vocabulary and a residual stream vector for any intermediate layer.

Now consider an actual activation $h_{(\ell)}^{tm}$. Its J-Lens score for token $v$ is, before normalization,

$$
h_{(\ell)}^{tm}
J_{(\ell)m}{}^n
W_n{}^v
$$

or, using $\widetilde J$,

$$
h_{(\ell)}^{tm}
\widetilde J_{(\ell)m}{}^v.
$$

This second form makes the interpretation especially clear. Suppose we fix a particular position $t$ and choose a vocabulary token $v$ while inspecting a particular layer $\ell$. Then, the above expression is simply the inner product between:

- the activation currently present at layer $\ell$, and
- the layer-local direction that tends to become downstream evidence for token $v$ (i.e., the J-vector).

A large score means that the activation is strongly aligned with a direction that the model generally transforms into evidence for saying that token.

The interpretation is not necessarily:

> The model is about to output “bug.”

It is closer to:

> The current activation contains a direction that, across many contexts, contributes to the model verbalizing “bug” eventually.

This is how the J-Lens turns a causal sensitivity into a human-interpretable window on a local activation. Every vocabulary item supplies a readable label, while the Jacobian ensures that the corresponding direction is expressed in the native coordinates of the layer being examined.

The real J-Lens also applies the model’s final normalization before unembedding, so it is written as follows:

$$
\operatorname{softmax}
\left(
W_U\,\operatorname{norm}(J_\ell h_\ell)
\right).
$$

Up to this activation-dependent normalization, Anthropic likewise describes the scores as inner products between the activation and token-specific J-Lens vectors.

## Conclusion

The whole argument can be summarized in three equations:

$$
\delta h_{(f)}
\approx
J\,\delta h_{(\ell)},
$$

$$
\delta z
\approx
W_U J\,\delta h_{(\ell)},
$$

and, after averaging over contexts and using the resulting directions as probes,

$$
\operatorname{score}_v(h_{(\ell)})
\approx
\left\langle
h_{(\ell)},
\text{average layer-local direction for token }v
\right\rangle.
$$

That is why the J-Lens expression should not feel arbitrary. If we want a local, layer-specific, human-readable account of what an activation is poised to do, the natural ingredients are:

- the activation itself;
- the derivative of the downstream model;
- the model’s own vocabulary readout; and
- an average that retains general verbalizable structure rather than the routing peculiarities of one prompt.

Their composition is precisely the J-Lens.
