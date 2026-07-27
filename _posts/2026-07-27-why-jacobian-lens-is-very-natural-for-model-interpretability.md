---
layout: post
title: "Why Jacobian Lens is very natural for model interpretability"
description: "How Jacobian Lens follows naturally from first-order calculus, the chain rule, and the model's own vocabulary readout."
date: 2026-07-27
math: true
---

When I first encountered the Jacobian lens in Section 2.1 of Anthropic’s paper, its central expression felt somewhat unmotivated:

$$
\operatorname{softmax}
\left(
W_U\,\operatorname{norm}(J_\ell h_\ell)
\right)
$$

Why should multiplying an intermediate activation by an averaged Jacobian, and then applying the model’s unembedding matrix, reveal anything meaningful about what the model is representing?

The answer is that this expression does not come out of nowhere. It follows naturally from one basic interpretability question:

> If we slightly perturb the residual stream at a particular layer, how does that perturbation affect what the model may eventually say?

Once we formulate the problem this way, the structure of the J-lens follows almost immediately from first-order calculus and the chain rule. Averaging then turns a context-specific derivative into a general-purpose reading tool.

## The minimum transformer background

A transformer processes text through a sequence of layers. At every token position, each layer maintains a vector called the residual stream. We can think of this vector as the model’s evolving internal state at that position.

Let

$$
h_{(\ell)}^{tm}
$$

denote the residual-stream activation at layer $\ell$ and token position $t$. The index $m$ selects one of the coordinates in the residual stream out of $d_{\mathrm{model}}$ total (this number is known usually as the "model dimension").

By the final layer, the residual stream has been transformed into a representation from which the model can predict tokens. Ignoring normalization and biases, the logit for vocabulary token $v$ at position $s$ is

$$
h_{(f)}^{sn}W_n{}^v
$$

For clarity, I define the following:

- $h_{(f)}^{sn}$ is the final-layer residual stream tensor.
- $W_n{}^v$ is the unembedding tensor.
- $v$ indexes vocabulary items.
- $m$ and $n$ are indices for residual-stream coordinates at an intermediate layer $\ell$ and final layer $f$, respectively.
- $t$ and $s$ index token positions at an intermediate layer $\ell$ and final layer $f$, respectively.

The superscripts are indices, not exponents. Under Einstein summation notation, an index that appears once above and once below is implicitly summed. Thus, the previous expression means we sum over the residual stream coordinates between the residual stream and the mapping between residual stream coordinates and vocabulary tokens the unembedding tensor provides:

$$
\sum_n h_{(f)}^{sn}W_n{}^v.
$$

This results in a final tensor of shape $(s,v)$ that maps from positions to vocabulary items. If this notation is unfamiliar, the central idea is simpler than the symbols: the final residual vector is multiplied by the unembedding matrix to produce one score for every possible output token.

## Begin with a perturbation

Suppose we make a small change

$$
\delta h_{(\ell)}^{tm}
$$

to the residual stream at layer $\ell$ and position $t$. That change propagates through the remaining layers and may alter the final residual stream at the same or a later position $s\ge t$.

To first order,

$$
\delta h_{(f)}^{sn}
\approx
\delta h_{(\ell)}^{tm}
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}.
$$

The derivative

$$
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}
$$

is a Jacobian. It describes how each coordinate of the earlier residual vector affects each coordinate of the final residual vector.

The corresponding change in the output logits is

$$
\delta h_{(f)}^{sn}W_n{}^v.
$$

Substituting the first-order propagation equation gives

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

This is already almost the complete J-lens expression.

It contains, in order:

1. The perturbation at the layer we want to interpret.
2. The Jacobian describing how that perturbation propagates downstream.
3. The unembedding direction describing how the final residual stream affects vocabulary token $v$.

By the chain rule,

$$
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}
W_n{}^v
$$

So composing the Jacobian with the unembedding does something remarkably concrete: it gives the direction, in the coordinate system of layer $\ell$, along which a small movement would increase or decrease the later logit for token $v$.

This is sometimes described as pulling the token’s unembedding direction backward through the model. Instead of assuming that the word “bug,” for example, has the same vector representation at every layer, we ask the model itself:

> Which direction at this particular layer tends to become downstream evidence for saying “bug”?

## Why average over contexts?

The Jacobian above is calculated during one particular prompt, between particular source and target positions. It therefore reflects both a general relationship and the accidental details of that context.

A representation of a bug might be used differently while reading code than while reading about insects. Attention patterns, surrounding concepts, and the model’s current task can all change how the representation is routed downstream.

Anthropic therefore averages the Jacobian over prompts, source positions, and present or future target positions:

$$
\mathbb E_{c,t,s\ge t}
\left[
\frac{\partial h_{(f)}^{sn}}
{\partial h_{(\ell)}^{tm}}
\right].
$$

Here $c$ indexes the prompt or context. After averaging, the prompt and position indices disappear, leaving one

$$
d_{\mathrm{model}}\times d_{\mathrm{model}}
$$

matrix for each layer.

This averaging does not make the result completely independent of context. It makes it representative of the distribution of contexts used to construct it. Context-specific effects that vary from example to example tend to weaken, while stable relationships reinforce one another.

The result answers a more general question:

> Across many situations, how does a direction at layer $\ell$ tend to affect the final residual stream now or later?

Anthropic describes this as separating a representation’s general disposition to be verbalized from the particular use being made of it in one prompt. Their implementation averages over source positions, later positions, and one thousand pretraining-like prompts. [Anthropic’s J-lens construction](https://transformer-circuits.pub/2026/workspace/index.html#the-jacobian-lens)

## From causal sensitivity to a reading tool

We can now compose the averaged Jacobian with the unembedding direction for token $v$:

$$
J_{(\ell)}^n{}_mW_n{}^v.
$$

For each vocabulary item $v$, this produces a direction

$$
\widetilde W_{(\ell)m}{}^v
$$

in the residual-stream space of layer $\ell$. It is the average, layer-local direction associated with increasing future evidence for that token. In Anthropic's original paper, it is exactly this (m, v)-shape tensor they term the "J-vector" - an explicit mapping between any token in a model's vocabulary and a residual stream vector for any intermediate layer.

Now consider an actual activation $h_{(\ell)}^{tm}$. Its J-lens score for token $v$ is, before normalization,

$$
h_{(\ell)}^{tm}
J_{(\ell)}^n{}_m
W_n{}^v
$$

or, using the pulled-back token direction,

$$
h_{(\ell)}^{tm}
\widetilde W_{(\ell)m}{}^v.
$$

This second form makes the interpretation especially clear. It is an inner product between:

- the activation currently present at layer $\ell$, and
- the layer-local direction that tends to become downstream evidence for token $v$ (i.e., the J-vector).

A large score means that the activation is strongly aligned with a direction that the model is generally capable of transforming into evidence for saying that token.

The interpretation is not necessarily:

> The model is about to output “bug.”

It is closer to:

> The current activation contains a direction that, across many contexts, is capable of contributing to the model verbalizing “bug.”

This is how the J-lens turns a causal sensitivity into a human-interpretable window on a local activation. Every vocabulary item supplies a readable label, while the Jacobian ensures that the corresponding direction is expressed in the native coordinates of the layer being examined.

The real J-lens also applies the model’s final normalization before unembedding:

$$
\operatorname{softmax}
\left(
W_U\,\operatorname{norm}(J_\ell h_\ell)
\right).
$$

Up to this activation-dependent normalization, Anthropic likewise describes the scores as inner products between the activation and token-specific J-lens vectors. [J-lens readout details](https://transformer-circuits.pub/2026/workspace/index.html#technical-details-of-j-lens-use-cases)

## Why this is such a natural choice

The ordinary logit lens applies the final unembedding matrix directly to an intermediate activation. In effect, it assumes that the model uses the same representational coordinates at every layer.

The J-lens removes that assumption. It asks how directions at the layer under inspection actually propagate toward the output, then uses that relationship to translate final-layer vocabulary directions back into the layer’s own coordinate system.

The whole argument can therefore be summarized in three equations:

$$
\delta h_{(f)}
\approx
J\,\delta h_{(\ell)},
$$

$$
\delta z
\approx
W_UJ\,\delta h_{(\ell)},
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

That is why the J-lens expression should not feel arbitrary. If we want a local, layer-specific, human-readable account of what an activation is poised to do, the natural ingredients are:

- the activation itself;
- the derivative of the downstream model;
- the model’s own vocabulary readout; and
- an average that retains general verbalizable structure rather than the routing peculiarities of one prompt.

Their composition is precisely the J-lens.
