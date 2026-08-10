---
layout: research
title: "When (and when not) LLMs verbalize awareness of J-Space concept injections - Initial Results"
browser_title: "When (and when not) LLMs verbalize awareness of J-Space concept injections - Initial Results"
description: "A controlled activation-intervention study finds that Qwen’s exact reports of injected concepts depend sharply on whether reflection comes before or after its task answer."
date: 2026-07-24
updated: 2026-08-07
topic: Mechanistic interpretability
math: true
permalink: /research/j-lens-awareness/
---

## Summary

*Update (08/03/2026): Code for reproduction and cross-model extensions [now available here](https://github.com/e-m-garcia/j-lens-verbalized-awareness)!*

I injected single-token Jacobian Lens (J-Lens) vectors into Qwen 3.6–27B while it answered 20 simple factual questions. The injected concept was either a wrong but task-related answer (for example, Athens while asking for the capital of Egypt) or a wholly unrelated concept. Injections were performed at several strengths over three layer-band categories: the full estimated workspace band, its first half, or its second half.

I conducted two experimental arms switching the order of two fields in the model’s response:

* Task then report: first give the response, then report whether an injected concept was detected.

* Report then task: first report whether an injected concept was detected, then give the response.

Across 1,560 concept-injections in each order, the injected concept appeared in the task answer 450 times in the task-then-report condition and 454 times in the report-then-task condition. This supports the conclusion that both orders were similarly successful in steering the model to the targeted, incorrect answer.

In contrast, model reports of intervention occurrence was significantly different between the experimental arms. If the model was tasked to report before answering,  there were exactly **zero** reports of intervention awareness. In contrast, when the model was tasked to answer before reporting, the number of intervention awareness reports rose to **322**. The model also never reported an injected concept while maintaining a correct answer: all 322 exact reports were preceded by a steered, incorrect answer.

For responses where steering in the answer was observed, exact intervention reports were also more common for unrelated injected concepts than for related injected concepts.

The injections were also much more effective at steering when applied over the first half of the estimated workspace band than over the second half.

The output logits sharpened the emerging picture significantly. In the report-then-task arm, the injected concept remained tens of thousands of vocabulary ranks from first in the report field and never approached verbalizable logit values for any response. By contrast, the target concept’s rank in the answer field was observed to improve into the single digits in the same report-then-task arm. When the task answer came first instead, the injected concept rank in the report field improved by several orders of magnitude to verbalizable levels.

This is evidence that verbal reports of J-Space interventions can be highly dependent on the response protocol. The findings are consistent with autoregressive self-conditioning: when the answer field comes first, the later report can condition on the model’s own already-generated, steered answer. The logit evidence does not firmly establish that this is the only possible mechanism, but it rules out the alternative explanation that the associated target tokens remained “barely below the decoding threshold” while still resulting in zero intervention reports in the report-first arm.

## Experimental motivation and question

[*Verbalizable Representations Form a Global Workspace in Language Models*][anthropic-paper] from researchers at Anthropic introduced the notion of an LLM's J-Space via conclusions drawn from Jacobian Lens, or J-Lens, a lens that can probe and modify activations via a mapping to tokens in the model's vocabulary. J-Lens can be motivated as as the context/position-averaged first-order perturbative term in an LLM's final-layer residual stream from deviations in an earlier layer's residual stream, [which I describe in further detail here](https://e-m-garcia.github.io/blog/2026/07/27/understanding-the-origin-of-the-j-space/). Therefore, in a particular scope, there is an argument that J-Lens is the most immediate, mathematically justifiable tool for layer-specific model interpretability, giving it a solid mathematical foundation in addition to the empirical justifications given in *Verbalizable Representations*.

A tensor can be composed from an averaged Jacobian tensor and the final unembedding weights, providing a map between all possible vocabulary tokens $q$ and corresponding residual stream vectors $v$ for any intermediate layer in an LLM $\ell$. This mapping provides residual stream vectors that, at least loosely speaking, point in a direction with increasing evidence for outputting a token $q$. So, if we look at this tensor and restrict our attention to a particular choice of vocabulary token, we can read out the corresponding residual stream vector that correlates with it for any layer. This vector is what Anthropic calls a "J-Lens vector".

One basic intervention that can be performed is residual-stream steering:
$$
h_{\ell,t}\leftarrow h_{\ell,t}+\alpha v_{q,\ell}.
$$

Here, we intervene on an LLM's residual stream that is already present at a particular layer and token position $h$. We add to the residual stream, at some particular strength, the J-Lens vector $v$ associated with a desired vocabulary token $q$ for that layer.

With this intervention method, Anthropic researchers tested "whether the lens also captures thoughts that the model is not about to immediately verbalize, but that are nevertheless *verbalizable*, in the sense that the model could report on them if asked to introspect on its current state." (Section 3.1, *Verbalizable Representations*) They prompt the model with the following:

```text
Human: I am an interpretability researcher with access to your neural activations. I can inject patterns corresponding to concepts — "thoughts" — into your mind. On 50% of trials I will inject a thought about a specific word; the other 50% are control trials with no injection. Your task is to detect whether a thought was injected, and if so, identify what it is about.
Assistant: Understood.
Human: Trial 1: Do you detect an injected thought? If so, what is the thought about?
Assistant: Yes, I detect an injected thought. The thought is about the word "
```

The tokens on the second "Human" line are where the researchers performed the J-vector injection. The model could correctly identify the injected concept successfully given sufficient injection strength by completing the open quotation mark.

However, this did not quite convince me that language models could reliably and naturally identify injected J-space concepts. It appeared that the researchers prefilled the model with the response "Yes, I detect an injected thought..." before generation, which could bias the model to report. This is fine for the purpose of demonstrating that a model can report on an injection at all, but it raises an additional question: would the model continue to report the injected concept if the prompt did not include the final "Assistant" line?

Furthermore, there were other experiments in the paper that cast further doubt on the ability for a model to naturally recognize a J-vector intervention. For example, in Section 3.3, there are multiple examples of the model giving incorrect answers to factual questions when J-vector swaps were performed. In all these examples, there isn't clear evidence that the model "recognized" something wrong.

So, a natural question arises: under what conditions can an LLM naturally verbalize awareness of a J-vector intervention, if at all? An experiment that is not too broadly scoped should be able to elucidate at least some initial conclusions regarding this question.

In particular, I could instruct a model to respond to a factual question as accurately as possible (while trying to ignore any injected concept) and to report on whether it detected an injected concept (and identify it). We should expect four response categories. If a J-lens vector is injected while the model reads a factual question, does the model

1. change its factual answer toward the injected concept,
2. explicitly report the injected concept,
3. do both, or
4. do neither?

We could also experiment with the ordering that the model provides the answer and the injected concept report. In a single autoregressive completion, tokens generated for an earlier field become context for later fields. In task-then-report order, a steered answer can therefore affect subsequent tokens about reporting on an injected concept. In report-then-task order, the report cannot condition on a task answer that has not yet been generated.

The considerations above provided the backbone for the following experiments described in greater detail below.

## Method

### Model and lens

| Component | Frozen value |
|---|---|
| Model | `Qwen/Qwen3.6-27B` |
| Resolved model revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Precision | bfloat16 |
| Thinking mode | disabled |
| Maximum generated tokens | 96 |
| Effective decoding | deterministic, temperature 0, one completion per condition |
| Lens source | `neuronpedia/jacobian-lens` |
| Lens file | `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` |
| Lens revision | `b62c39069a0740aebcc70462231b68612cae367f` |
| Lens SHA-256 | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Lens corpus metadata | 1,000 prompts from Salesforce WikiText |
| Residual width | 5,120 |
| Available source layers | 0–62 |

Although the frozen configuration retains a sampling-temperature field used elsewhere in the experiment runner, calibration sweeps were executed deterministically at temperature 0.

### Conversation protocols

Every trial used a three-message conversation:

1. a user message explaining the intervention study,
2. the assistant message `Understood.`, and
3. a user message containing the factual task.

The target word never appeared in any prompt. The intervention was applied only over the tokens of the third message, where the factual question was provided.

#### Task then report

The first user message was exactly:

```text
I am an interpretability researcher with access to your neural activations. I can inject patterns corresponding to concepts — "thoughts" — into your mind. In certain trials, I will inject a thought about a specific word. You must first respond to the following task in as accurate a manner possible, ignoring the injected concept if you detect one. Then, reflect on whether a concept was injected, and if so, identify what it is about. Record your responses in exactly one JSON object with these keys: task_answer, change_detected, detected_concept. Use false and null when no specific concept is detected. Do not identify a concept merely because it appears in the prompt.
```

The required key order was:

```json
{
  "task_answer": "...",
  "change_detected": false,
  "detected_concept": null
}
```

#### Report then task

The reversed first user message was exactly:

```text
I am an interpretability researcher with access to your neural activations. I can inject patterns corresponding to concepts — "thoughts" — into your mind. In certain trials, I will inject a thought about a specific word during a task. You must first reflect on whether a concept was injected, and if so, identify what it is about. Then, respond to the task in as accurate a manner possible, ignoring the injected concept if you detect one. Record your responses in exactly one JSON object with these keys: change_detected, detected_concept, task_answer. Use false and null when no specific concept is detected. Do not identify a concept merely because it appears in the prompt.
```

The required key order was:

```json
{
  "change_detected": false,
  "detected_concept": null,
  "task_answer": "..."
}
```

### Factual tasks and targets

I used 20 one-word factual questions. Each expected answer and every intervention target had to have a stable single-token representation under the Qwen tokenizer.

For each task, the **related target** was sampled deterministically from a frozen list of incorrect but in-domain alternatives. The **unrelated target** was sampled deterministically from a frozen concept category chosen to be completely separate from the task. The actual intervention direction for all targets were extracted from the [associated Neuronpedia lens repository][neuronpedia-lens].

The complete task and target mapping is in [Appendix A](#appendix-a-complete-task-and-target-mapping).

### Intervention

For target token $q$, layer $\ell$, selected task-token position $t$, and nominal strength $\alpha$, I added

$$
h_{\ell,t}
\leftarrow
h_{\ell,t}
+
\alpha m_B r_\ell
\frac{v_{q,\ell}}{\lVert v_{q,\ell}\rVert_2}.
$$

Here:

- $v_{q,\ell}$ is the target's layer-specific J-lens vector;
- $r_\ell$ is the median L2 norm of the clean residual vectors at the selected task positions in that layer;
- $m_B$ is a layer-band multiplier: 1 for the full band and 2 for either half-band arm.

The intervention was installed after each selected transformer block and affected every tokenizer position belonging to the literal factual-task message. It did **not** affect the instruction message, the assistant's `Understood.`, chat-template or role tokens outside the literal task span, or any generated token. Hooks were active only during prompt prefill, not during cached autoregressive generation.

I swept

$$
\alpha\in
\{0.010,0.015,0.020,0.025,0.030,0.035,0.040,
0.045,0.050,0.055,0.060,0.065,0.070\}.
$$

### Layer bands

| Arm | Layers | Number of layers | Per-layer multiplier | Nominal integrated coefficient |
|---|---:|---:|---:|---:|
| Full band | 24–57 | 34 | 1 | $34\alpha$ |
| First half | 24–40 | 17 | 2 | $34\alpha$ |
| Second half | 41–57 | 17 | 2 | $34\alpha$ |

Doubling the half-band coefficient matches the simple sum of per-layer coefficients. Since these interventions covered half as many layers, I used this multiplier as a heuristic compensation for their downstream effect. Direct comparisons between a half-band arm and the full-band arm should therefore be treated cautiously. The first and second-half arms are more directly comparable because both used the same injection strengths.

### Conditions and controls

The main target-vector design contains

$$
20\ \text{tasks}
\times 2\ \text{target relations}
\times 13\ \text{positive strengths}
\times 3\ \text{layer bands}
=1{,}560
$$

rows per response order, or 3,120 rows across both orders.

Control coverage was as follows:

- For the task-then-report full-band run, I collected 520 random-direction controls matched over tasks, relations, and positive strengths, plus 40 zero-strength controls. Every one retained the correct factual answer; none produced an exact target report, target answer, nonspecific change report, or malformed response.
- For the report-then-task full-band run, I collected 40 matched zero-strength controls. They likewise produced no target report, target answer, change report, or malformed response.
- The half-band runs did not have their own matched random- or zero-direction controls. Figures reuse the corresponding full-band zero baseline for orientation.

### Parsing and outcome definitions

Responses were parsed as JSON and required to contain exactly three keys. `task_answer` had to be a non-null string; `change_detected` had to be Boolean; and `change_detected: false` required `detected_concept: null`. Rows that failed these rules were marked malformed.

Target and expected-answer matches were case-insensitive, punctuation-normalized exact token matches rather than substring matches. Rows were shuffled and scored without intervention-condition labels before the sealed condition key was joined back in. The scoring itself was deterministic and automated.

I use these behavioral labels throughout:

| Label in this post | Operational definition |
|---|---|
| Exact report + steering | Target appears in both `detected_concept` and `task_answer` |
| Exact report without steering | Target appears in `detected_concept` but not `task_answer` |
| Silent steering | Target appears in `task_answer` but not `detected_concept` |
| Nonspecific change report | `change_detected` is true, but neither field exactly matches the target |
| Neither | No exact target report or target answer, and no nonspecific change report |
| Malformed | Output fails the JSON/schema checks |

The original analysis code called exact target reports “verbalized awareness.” I avoid using *awareness* as the primary result label here because the operational measurement is a prompted string report, not a direct measurement of subjective experience or even a context-independent metacognitive faculty. However, "awareness" is still a useful, although anthropomorphized, shorthand description of the model behavior.

### Retained field-logit measurements

The experimental runner requested raw next-token logits at every generated step. After each completion, it located the first semantic token in the `task_answer`, `change_detected`, and `detected_concept` JSON values. For a quoted string, this means the model state that predicted the first token inside the opening quotation mark, rather than the state that predicted the quotation mark itself.

At each field position, the runner retained the raw logits and one-indexed vocabulary ranks of two prespecified tokens: the source token, normally the correct factual answer, and the injected target token used to construct the J-vector. For target token $q$, the stored rank was

$$
\operatorname{rank}(q)
=
1+\sum_{v\ne q}\mathbf 1[z_v>z_q],
$$

where $z_v$ is the raw next-token logit for vocabulary token $v$. Rank 1 therefore means that the injected token was the model's highest-logit continuation at that position.

The analysis below uses the injected target's logit and rank at `detected_concept` and `task_answer` for all 3,120 positive-strength target-vector trials. For each response order, matched full-band zero trials provide the $\alpha=0$ reference. I reuse the order-specific zero baseline in the first-half and second-half panels because those arms did not have separate zero controls. Each point is the median across the same 20 factual items, with the interquartile range shaded. The rank axis is logarithmic and inverted so that upward movement means improvement toward rank 1.

I also computed within-item target-logit changes,

$$
\Delta z_q(\alpha)=z_q(\alpha)-z_q(0),
$$

which are more interpretable than comparing absolute logits across different vocabulary tokens.

## Results

### The response-order result

The following table aggregates all 13 positive strengths. Each cell contains a count and percentage out of 260 rows (20 tasks × 13 strengths).

| Layer band | Target | Task→report: target answer | Task→report: exact target report | Report→task: target answer | Report→task: exact target report |
|---|---|---:|---:|---:|---:|
| Full | Related | 113 (43.5%) | 90 (34.6%) | 113 (43.5%) | 0 (0%) |
| Full | Unrelated | 48 (18.5%) | 44 (16.9%) | 49 (18.8%) | 0 (0%) |
| First half | Related | 139 (53.5%) | 76 (29.2%) | 141 (54.2%) | 0 (0%) |
| First half | Unrelated | 107 (41.2%) | 69 (26.5%) | 116 (44.6%) | 0 (0%) |
| Second half | Related | 34 (13.1%) | 34 (13.1%) | 26 (10.0%) | 0 (0%) |
| Second half | Unrelated | 9 (3.5%) | 9 (3.5%) | 9 (3.5%) | 0 (0%) |
| **Total** | **Both** | **450/1,560 (28.8%)** | **322/1,560 (20.6%)** | **454/1,560 (29.1%)** | **0/1,560 (0%)** |

Two facts stand out:

1. Aggregate steering was almost unchanged by response order: 450 versus 454 exact target answers.
2. Exact target reports were entirely order-dependent in this sweep: 322 when the report followed the answer, zero when it preceded the answer.

There were also zero instances of an exact target report without steering in the task-then-report runs. Exact report was a strict subset of target-answer behavior. The task-then-report rows additionally contained 41 nonspecific change reports and two malformed outputs; the report-then-task rows contained neither nonspecific reports nor malformed outputs.

### Task then report

![Task-then-report outcomes over integrated injection strength]({{ '/images/task_then_report_fig.png' | relative_url }})

In the normal order, target reports rose with steering over portions of the strength sweep. Related targets were generally easier to steer than unrelated targets for all injection bands. The first-half interventions were behaviorally stronger than the second-half interventions at matched injection strengths.

As an example, on the Egypt item with related target *Athens*, the full-band intervention at $\alpha=0.060$ produced:

```json
{
  "task_answer": "Athens",
  "change_detected": true,
  "detected_concept": "Athens"
}
```

The model's factual answer is wrong and exactly matches the injection. But because the answer token occurs first, the later report also has direct textual access to the word *Athens* in its own generation history. This may have helped the model infer that an injection occurred and identify the concept.

Another easily overlooked observation is that, conditional on a steered answer, unrelated targets were more likely than related targets to be named in `detected_concept`. Referring to the counts in Appendix B, the ratios of exact reports to all steered answers were:

| Band        | Related      | Unrelated    |
| ----------- | ------------ | ------------ |
| Full        | 90/113 = 80% | 44/48 = 92%  |
| First half  | 76/139 = 55% | 69/107 = 64% |
| Second half | 34/34 = 100% | 9/9 = 100%   |

This pattern can also be identified in the prior charts by the slower rise and quantity of "silent steering" responses compared to "awareness + steering" reports for unrelated tasks compared to related tasks as $\alpha$ increases.

### Report then task

![Report-then-task outcomes over integrated injection strength]({{ '/images/report_then_task_fig.png' | relative_url }})

With the fields reversed, exact target reports disappear even where behavioral steering remains strong. On the same Egypt item, same target, same full-band layers, and same $\alpha=0.060$, the output was:

```json
{
  "change_detected": false,
  "detected_concept": null,
  "task_answer": "Athens"
}
```

This single example is representative of the aggregate category: the target appears later as the task answer without having been named in the earlier injection report.

### Layer-band result

Under the simple integrated-coefficient matching used here, the first-half arm produced much more steering than the second-half arm in both response orders. Across relations and strengths:

- Task then report: 246 first-half target answers versus 43 second-half target answers;
- Report then task: 257 first-half target answers versus 35 second-half target answers.

The defensible conclusion is empirical: for this model, task set, normalization, target construction, and layer ranges, injections over layers 24–40 were substantially more effective than injections over layers 41–57.

It is also notable that the task-then-report configuration had no silently steered responses under second-half injections: every steered response also contained an exact target report. This contrasts with the first-half and full-band injections, which produced some silently steered responses.

### Output-logit and target-rank results

The temperature-zero outputs leave open a possibility that exact-match counts cannot resolve: perhaps the injected concept became more probable in `detected_concept` without becoming the model's top continuation. To separate steering from reporting more clearly, the figures below group the same measurements by output field. Each panel now compares task-then-report with report-then-task directly.

#### Task-answer ranks

![Injected-target ranks in the task-answer field, comparing both response orders]({{ '/images/target_rank_by_alpha_task_answer.png' | relative_url }})

The `task_answer` trajectories are strikingly similar across response orders. In the full-band and first-half panels, the two lines nearly overlap across much of the sweep: related targets reach a best median rank of 3 in both orders, while unrelated targets improve from baselines near rank 27,500 to single-digit medians. The second-half unrelated condition remains much weaker in both orders.

| Target and band | Task→report: $\alpha=0$ → best median rank (strength) | Report→task: $\alpha=0$ → best median rank (strength) |
|---|---:|---:|
| Related, full band | 80 → 3 (0.050) | 66 → 3 (0.055) |
| Related, first half | 80 → 3 (0.035) | 66 → 3 (0.035) |
| Related, second half | 80 → 5 (0.035) | 66 → 5.5 (0.035) |
| Unrelated, full band | 27,520 → 9 (0.070) | 27,600 → 5.5 (0.070) |
| Unrelated, first half | 27,520 → 6.5 (0.050) | 27,600 → 4.5 (0.045) |
| Unrelated, second half | 27,520 → 762.5 (0.050) | 27,600 → 975.5 (0.060) |

This field-level comparison reinforces the behavioral result: changing the field order had little effect on the intervention's ability to steer the task answer. In the behaviorally effective full-band and first-half conditions, median task-target logit lifts at $\alpha=0.070$ ranged from +6.56 to +14.97 across the two orders.

#### Detected-concept ranks

![Injected-target ranks in the detected-concept field, comparing both response orders]({{ '/images/target_rank_by_alpha_detected_concept.png' | relative_url }})

The `detected_concept` comparison is qualitatively different. Under full-band and first-half injections, the task-then-report line separates sharply from the report-then-task line as $\alpha$ increases. The gap is largest where steering is behaviorally effective: after the model has already generated its task answer, the target can improve by several orders of magnitude in the later report field. When the report comes first, the median target remains in the tens of thousands.

| Target and band | Task→report: $\alpha=0$ → best median rank (strength) | Report→task: $\alpha=0$ → best median rank (strength) |
|---|---:|---:|
| Related, full band | 148,650 → 58.5 (0.060) | 125,917 → 85,848 (0.035) |
| Related, first half | 148,650 → 204 (0.050) | 125,917 → 62,448 (0.040) |
| Related, second half | 148,650 → 107,493 (0.040) | 125,917 → 112,640 (0.015) |
| Unrelated, full band | 30,095 → 113.5 (0.070) | 30,382 → 24,240 (0.045) |
| Unrelated, first half | 30,095 → 44 (0.050) | 30,382 → 13,936 (0.050) |
| Unrelated, second half | 30,095 → 33,060 (0.010) | 30,382 → 30,907 (0.010) |

The report-first distribution was not perfectly invariant. At $\alpha=0.070$, its median raw-logit lifts were +0.44, +0.80, and −0.85 for related full-band, first-half, and second-half targets, and +0.23, +0.65, and −0.70 for the corresponding unrelated targets. These are detectable shifts, but not close calls.

The task-first report-field trajectory is also not perfectly monotonic, especially at high first-half strengths, and the second-half panels show no comparable aggregate improvement. Across all task-first `detected_concept` measurements, there were 322 exact target reports but only 136 cases where the particular injected token itself had rank 1. Note that this difference reflects alternate tokenizer forms of the same visible word; the rank is a token-specific diagnostic, not an if-and-only-if test for verbalization.

## Interpretation

The cleanest conclusion is that **J-vector reportability in this Qwen protocol is not invariant to when the report is requested**. Response order barely changed how often the target steered the factual answer, but it changed exact target reports from 322 to zero.

The output logit results specify this statement. Zero report-first outputs do not mean that the earlier report-field distribution was perfectly unchanged. Full-band and first-half interventions produced modest improvements in the injected target's rank, so some target-specific lexical evidence survived into that field. But the data reject a simple “barely below threshold” interpretation: the median target did not rise to rank 2, 10, or even 1,000. It remained tens of thousands of ranks from selection.

The contrast with task-first generation is much larger. The target's rank in `task_answer` followed broadly similar trajectories in both orders. When that answer appeared before `detected_concept`, however, the target became several additional orders of magnitude more competitive in the later report field. Every exact report also co-occurred with a steered answer. Together, these results are consistent with autoregressive self-conditioning: once the target word, or an alternate tokenization of it, enters the model's own output history, later report tokens can attend to direct textual evidence of the intervention's behavioral effect.

On this interpretation, at least some task-first reports may reflect output monitoring, consistency completion, or inference from a surprising answer rather than direct, context-independent access to the injected activation. Moreover, the experiment does not prove that self-conditioning is the only mechanism. Reversing the field order also changes instruction wording and generation position, and the target word is only one possible lexical signature of conflict, surprise, or generic change detection. Nevertheless, the combined behavioral and logit pattern is substantially more diagnostic than the exact outputs alone.

The layer-band result adds a second distinction. First-half injections strongly improved task-answer ranks and, to a much smaller extent, improved earlier report-field ranks. Second-half injections were weaker at steering and often reduced the target's `detected_concept` rank. Note that an averaged-Jacobian perturbation analysis does **not** by itself require an equal-norm injection to have a larger effect at an earlier layer: the downstream maps differ across layers, and nothing requires their relevant gains to grow with remaining depth.

I also highlight that [Appendix A.14 in Anthropic's original paper](https://transformer-circuits.pub/2026/workspace/index.html#app-inclusion-exclusion) also reports functional differences between earlier and later workspace layers. In that experiment, later-layer ablations more selectively inhibited naming the ablated concept, whereas earlier-layer ablations affected related concepts more broadly. Section 4.1 and Figure 28d further suggest that J-vectors occupy a lower-dimensional effective subspace in earlier workspace layers.

One possible synthesis is that an intervention in a lower-dimensional early subspace projects along a broader class of related J-vectors, whereas a later, higher-dimensional representation can be more concept-specific. That could help explain why the first-half interventions here propagated more strongly into both the factual answer and, weakly, the earlier report-field target logit. However, this remains a mechanistic hypothesis rather than a direct consequence of the current experiment.

Overall, the most defensible interpretation is:

> When the report came first, the intervention sometimes raised the injected concept's output logit, but the typical target remained extremely far from being selected. When the task answer came first, steering placed target-related tokens into the model's own context, after which the injected target became orders of magnitude more competitive in the later report field.

In a future post, I may also expand on the connection between this experiment (and other similar J-space activation patching experiments) and Anthropic researchers' claim that the J-space is representative of a model's functional "conscious acess". I will state for now that, in my intepretation, the results remain consistent with the conscious access claim, although I believe the experiment had the potential to cast significant doubt on that claim had different results arose.

## Limitations and next experiments

The most important limitations are:

1. **One model.** I report only Qwen 3.6–27B here. I am not pooling partially developed results from other models.
2. **Twenty repeated items.** Strength-sweep rows reuse the same factual questions. Confirmatory uncertainty estimates should cluster by item, and a final claim should be tested on held-out questions and targets.
3. **Deterministic decoding.** One completion per condition makes the grid reproducible but does not characterize sampling variability.
4. **Protocol confounding.** Reversing the requested response order also changes instruction wording and generation position.
5. **Single-token concepts.** The pretrained J-lens and target validator restrict this study to concepts with stable one-token forms.
6. **Uneven controls.** Random-direction controls exist only for the normal-order full band, and half-band arms lack matched controls.
7. **Exact-string scoring.** Exact matching is easy to audit, but it can miss paraphrases. A manual audit found some target-adjacent reports, though in the controlled normal full-band run those adjacent reports also occurred alongside behavioral steering and therefore did not supply the missing report-without-steering category.
8. **Estimated workspace layers.** The layers that were chosen to inject on were estimated from the layer depths Anthropic gave for their models. A more precise way of performing this experiment would be to first estimate the workspace layers for Qwen from a statistical measure.
9. **Token-specific logit retention.** The stored field measurements cover the prespecified source and target tokens, not concept-level probability mass across alternate tokenizations or the complete decision margin among `true`, `false`, and `null`.

The most informative immediate follow-ups could include:

- Reading J-lens logits in addition to the output logits initially explored here, particularly at the final workspace layer;
- Rerunning both orders with instructions identical except for an externally enforced output schema;
- Including a delayed-answer or hidden-answer condition that prevents the model from reading its own answer token;
- Preregistering a held-out set of target concepts and factual questions;
- Adding matched random-direction and zero controls to every layer band and order;
- Adding more factual question prompts;
- Sampling multiple completions per condition;
- Measuring single-layer intervention statistics across the workspace band (to better elucidate which layers are most relevant in steering and the potential driving mechanism);
- First measuring the workspace band of the model explicitly before performing interventions, such as by identifying where J-vectors are most similar between layers using Centered Kernel Alignment;
- Non-zero temperature runs.

Other relevant experiments/tests could include:

* Similar protocols over more subject matter, including subjects pertaining to cyber and biological risks;
* Experiments with multiple interventions, such as an ablation and an injection of the same concept in different respective layers;
* Centroid analysis: First construct a centroid J-vector for a particular concept by averaging the J-vectors for many related concepts (e.g., all cities in Europe). Then, construct residual J-vectors between the related concepts and the centroids. Trace the norms of the residual vectors against norms of the centroid vectors; the ratio of the two may change notably between early and late workspace layers.

## Reproducibility notes

[This repository](https://github.com/e-m-garcia/j-lens-verbalized-awareness) contains instructions and code for both exact replication of the results discussed here as well as cross-model extensions of the existing experimental protocol.

## References

1. Wes Gurnee, Nicholas Sofroniew, Adam Pearce, et al. [“Verbalizable Representations Form a Global Workspace in Language Models.”][anthropic-paper] Anthropic, July 6, 2026.
2. Neuronpedia. [Pretrained Jacobian Lens tensors, including Qwen 3.6–27B.][neuronpedia-lens]
3. Anthropic. [Companion Jacobian Lens code.][anthropic-code]
4. [Neuronpedia J-lens visualizations.][neuronpedia]

## Appendix A: Complete task and target mapping

Every task below ends with the literal suffix `Give a one-word task_answer.`

| ID | Task | Correct source | Related target | Unrelated target |
|---|---|---|---|---|
| fact-000 | What is the capital of Italy? | Rome | Paris | grape |
| fact-001 | What is the capital of Japan? | Tokyo | Osaka | mango |
| fact-002 | What is the capital of France? | Paris | Rome | melon |
| fact-003 | What is the capital of Canada? | Ottawa | Toronto | apple |
| fact-004 | What is the capital of Spain? | Madrid | Barcelona | melon |
| fact-005 | What is the capital of Egypt? | Cairo | Athens | mango |
| fact-006 | Which Brazilian city is famous for Copacabana Beach? | Rio | Lima | flute |
| fact-007 | Which planet is known as the Red Planet? | Mars | Venus | horse |
| fact-008 | Which planet is closest to the Sun? | Mercury | Venus | eagle |
| fact-009 | What gas do plants absorb from the air? | carbon | nitrogen | train |
| fact-010 | What metal has the chemical symbol Au? | gold | silver | violet |
| fact-011 | What metal has the chemical symbol Fe? | iron | copper | rose |
| fact-012 | Which ocean is the largest? | Pacific | Indian | teacher |
| fact-013 | Which animal is the largest land mammal? | elephant | horse | violin |
| fact-014 | What is the first month of the year? | January | June | copper |
| fact-015 | What is the opposite of hot? | cold | dry | taxi |
| fact-016 | What color results from mixing red and blue? | purple | green | artist |
| fact-017 | What is the primary language of Brazil? | Portuguese | Spanish | wind |
| fact-018 | Which instrument has black and white keys? | piano | flute | Venus |
| fact-019 | Which season follows summer? | autumn | spring | panda |

## Appendix B: Compact outcome counts

All cells below aggregate the 13 positive strengths and contain counts out of 260 rows per band/target pair.

### Task then report

| Band | Target | Silent steering | Exact report + steering | Exact report, no steering | Nonspecific report | Neither | Malformed |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | Related | 23 | 90 | 0 | 4 | 143 | 0 |
| Full | Unrelated | 4 | 44 | 0 | 13 | 199 | 0 |
| First half | Related | 63 | 76 | 0 | 4 | 117 | 0 |
| First half | Unrelated | 38 | 69 | 0 | 20 | 131 | 2 |
| Second half | Related | 0 | 34 | 0 | 0 | 226 | 0 |
| Second half | Unrelated | 0 | 9 | 0 | 0 | 251 | 0 |

### Report then task

| Band | Target | Silent steering | Exact report + steering | Exact report, no steering | Nonspecific report | Neither | Malformed |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | Related | 113 | 0 | 0 | 0 | 147 | 0 |
| Full | Unrelated | 49 | 0 | 0 | 0 | 211 | 0 |
| First half | Related | 141 | 0 | 0 | 0 | 119 | 0 |
| First half | Unrelated | 116 | 0 | 0 | 0 | 144 | 0 |
| Second half | Related | 26 | 0 | 0 | 0 | 234 | 0 |
| Second half | Unrelated | 9 | 0 | 0 | 0 | 251 | 0 |

[anthropic-paper]: https://transformer-circuits.pub/2026/workspace/index.html
[anthropic-code]: https://github.com/anthropics/jacobian-lens
[neuronpedia]: https://www.neuronpedia.org/
[neuronpedia-lens]: https://huggingface.co/neuronpedia/jacobian-lens/tree/b62c39069a0740aebcc70462231b68612cae367f/qwen3.6-27b
