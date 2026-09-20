*This project has been created as part of the 42 curriculum by romgutie.*

# call me maybe

## Description

This program turns natural language requests into structured function
calls. Given `What is the sum of 2 and 3?` it does not answer `5`, it
answers which function to call and with which arguments:

```json
{ "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2, "b": 3 } }
```

The model behind it is Qwen3-0.6B, which is far too small to be
trusted with producing valid JSON on its own. Nothing here relies on
it doing so: the JSON structure is never generated, it is imposed.
At every step the set of tokens that would keep the output valid is
computed, every other token is pushed to `-inf`, and the model only
picks inside what is left. Invalid output is therefore not unlikely,
it is unreachable.

## Instructions

```bash
make install                 # uv sync
make run                     # uv run python -m src
make lint                    # flake8 + mypy
```

Input files are read from `data/input/` and the result is written to
`data/output/function_calling_results.json`. Custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

A successful run prints nothing; errors go to stderr and the exit
code is `1`.

## Algorithm

One call is produced in two stages, both constrained.

**1. Choosing the function.** The declared names are tokenised and
walked together, token by token. At each step only the tokens that
continue at least one candidate name are allowed, so the model can
only ever spell a name that exists. No string matching, no keyword
heuristic: the choice comes from the logits.

**2. Filling the parameters.** Keys, braces, commas, colons and the
quotes around strings are written directly from the schema. Only the
values are generated, each by the generator of its declared type:

- *number* — digits only, one dot at most, never a trailing sign or
  dot. There is no "end of number" token, so the tokens that may
  follow the value (`,` `}`) are offered as a stop signal: choosing
  one ends the value without being consumed.
- *boolean* — a single token, `true` or `false`.
- *string* — two regimes in the same loop. While what has been
  written is the start of a passage of the request, the value is a
  copy: the only allowed token is the next one of that passage.
  Otherwise the value is written freely, restricted to the tokens
  whose text holds neither `"` nor `\`, so the string cannot be
  closed early nor hold an invalid escape. Closing works like it does
  for numbers: any token starting with a quote (`"`, `",`, `"}`) is a
  stop signal.

The name tokens stay in the context while the parameters are written,
so the values are conditioned on the function the model has just
committed to.

## Design decisions

**The model is only called when the constraint leaves a choice.** If
the allowed set holds a single token, that token is written without a
forward pass. This is what makes copying free: once the model has
started a passage of the request, the rest of it follows
deterministically. Copying an 18-character passage costs one forward
pass instead of one per token.

**Closing tokens are a set, not a single quote.** In real JSON the
closing quote is glued to what follows and the tokeniser makes `",`
or `"}` a single token. Allowing only the lone `"` hid the natural
way out and pushed the model to keep writing — this was the direct
cause of truncated, invalid patterns.

**Unsupported parameter types are rejected at load time.** A function
declaring an `array` or an `object` fails pydantic validation and is
dropped from the candidates, rather than producing a call with a
missing argument.

**Numbers are emitted as `int` when they have no decimal part.** Both
`2` and `2.0` are valid JSON numbers for a `number` parameter.

## Performance

| | before | after |
|---|---|---|
| total time, 11 prompts | 322 s | 85 s |

The gain comes almost entirely from the forward passes skipped when
the constraint is already decisive; the number of generated tokens
barely changed. The SDK exposes no KV cache, so each forward pass
recomputes the whole context — the only real lever is the number of
calls, not their cost.

Accuracy on the sample set: 11/11 function names, 24/25 arguments.
JSON validity: 100% by construction, on every run, whatever the model
produces.

## Challenges

**Single-character tokens destroy the model's distribution.** The
first version restricted strings to one-character tokens. It produced
garbage (`"g' 2023-04-15T00:00"`) because the model never sees text
tokenised that way during training. Allowing every safe token of the
vocabulary fixed it while keeping the same guarantee.

**A truncated value is a symptom, not the disease.** Patterns kept
being cut mid-expression (`([aeiou])|([aeiou`). The repetition guard
was firing correctly; the cause was upstream, in the missing closing
tokens described above.

**Copying versus inventing.** A string value is sometimes a passage
of the request and sometimes something the request only describes,
like a regex. When the request quotes a passage, only quoted passages
are copy candidates; otherwise words are. The known limitation: in a
request with no quotes, a value meant to be invented can be captured
by a word of the request.

**Naming a symbol.** `with asterisks` should yield `*`. The character
appears nowhere in the request, so no generic constraint can produce
it — only prompting can, and a 0.6B model does not always follow.

## Testing

The model is slow and non-deterministic to iterate against, so most
of the checking was done against a fake SDK returning random logits
over a small vocabulary. Two hundred generated calls confirmed that
the program never crashes, always terminates, and always produces
parameters matching the declared schema, whatever the model "says".
Scripted logits were then used to check single behaviours: that a
copy costs one forward pass, that a multi-character closing token
ends a value without polluting it, that a number never decodes from
an empty or dangling token. Edge cases on the real model: missing and
malformed input files, unsupported parameter types, empty prompts.

## Resources

- [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) — the JSON
  grammar the decoder enforces.
- [Efficient Guided Generation for Large Language
  Models](https://arxiv.org/abs/2307.09702) — Willard & Louf, the
  reference paper on constrained decoding.
- [Language Models are Unsupervised Multitask
  Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  — byte-level BPE, which explains why a space is `Ġ` in the
  vocabulary file.
- [Qwen3 documentation](https://qwen.readthedocs.io/) — chat
  template and the `<think>` block.
- [Hugging Face: generation
  strategies](https://huggingface.co/docs/transformers/generation_strategies)
  — how logits processors intervene before sampling.


### Use of AI

AI was used as a reviewer and debugging partner to identify bugs, refine the constrained decoding and prompts, and write temporary tests. All suggestions were reviewed, tested, and adapted where necessary.
