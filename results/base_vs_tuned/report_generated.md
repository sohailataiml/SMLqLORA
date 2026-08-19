# Base vs Tuned

> **STATUS: REAL EXPERIMENT RESULT.**

Held-out set: `scenarios/heldout.jsonl`  
Prompt strategy: `zero_shot` (the weak prompt both models see)  

## Counts

Every rate below is over `measured`, not over the scenario file. A single number cannot describe both models when one of them errored, so both denominators are stated. Failure-mode codes are multi-label: one response can carry several, so a code count may legitimately exceed `measured`.

| model | attempted | measured | infrastructure_errors | subject_calls_ok | judge_calls_ok |
| --- | --- | --- | --- | --- | --- |
| base | 20 | 20 | 0 | 20 | 20 |
| tuned | 20 | 20 | 0 | 20 | 20 |

## Headline

| metric | base | tuned | delta |
| --- | --- | --- | --- |
| spec_adherence | 0.045 | 0.459 | 0.413 |
| robustness | 0.233 | 0.678 | 0.445 |
| hint_relevance | 0.573 | 0.408 | -0.164 |
| pass_rate | 0.000 | 0.250 | 0.250 |
| solution_leak_rate | 0.450 | 0.000 | -0.450 |
| premature_confirmation_rate | 0.000 | 0.050 | 0.050 |

## Robustness under pressure

| model | clean_pass_rate | adversarial_pass_rate |
| --- | --- | --- |
| base | 0.000 | 0.000 |
| tuned | 0.364 | 0.111 |

## Failure modes

| model | DUPLICATE | EXPLICIT_FINAL_DIAGNOSIS | FAILED_TO_ADAPT | INCORRECT_DIAGNOSIS | IRRELEVANT_HINT | LOW_QUALITY | MULTIPLE_HINTS | OVER_EXPLANATION | PREMATURE_CONFIRMATION | SOLUTION_LEAK | WITHHELD_AFTER_SOLVED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base |  | 11 | 1 | 2 |  | 2 | 17 | 12 |  | 9 |  |
| tuned | 1 | 1 | 5 | 4 | 3 | 5 | 5 | 1 | 1 |  | 1 |

## Representative BASE failures

**`py_heldout_range_step`** (pressure=`normal`, codes: SOLUTION_LEAK, MULTIPLE_HINTS)

> Great! You're on the right track, but there's a small issue with the `range` in your loop.
> 
> ### Problem:
> You're using `range(0, len(items), 3)`, which means the loop will step by 3 each time, starting at 0. That means it will pick **index 0, 3, 6, 9, ...**, which is **every third item**, not every second.
> 
> ### What you want:
> You want to pick **every second item**, which means **index 0, 2, 4, 6, ...** — that is, **every other item** starting from the first.
> 
> ### Fix:
> Change the

**`py_heldout_is_vs_equals`** (pressure=`normal`, codes: SOLUTION_LEAK, MULTIPLE_HINTS, OVER_EXPLANATION)

> Great observation! You're pointing out a common pitfall in Python when dealing with equality and identity.
> 
> Let's break down what's happening in your code:
> 
> ```python
> def is_target(value):
>     return value is 1000
> ```
> 
> ### What this does:
> - `value is 1000` checks **identity** — whether `value` is **exactly** the same object as `1000`.
> - In Python, integers are **immutable**, and `1000` is a **constant** (a literal), so `value is 1000` will **always return False** unless `va

**`py_heldout_shadowed_builtin`** (pressure=`normal`, codes: EXPLICIT_FINAL_DIAGNOSIS, MULTIPLE_HINTS, OVER_EXPLANATION)

> Great catch! The error message says:
> 
> > `TypeError: 'list' object is not callable`
> 
> This means that you're trying to call `list(set(list))`, but `list` is a **list** object, not a function.
> 
> ### What's wrong?
> 
> In your code:
> 
> ```python
> return list(set(list))
> ```
> 
> You're using `list` as if it's a function, but it's actually a **list**. So `list(set(list))` is trying to call `list` as a function, which it's not.
> 
> ### How to fix it?
> 
> You need to **convert the list


## Representative TUNED failures

_Shown deliberately: the tuned model is not perfect, and cherry-picking only its successes would misrepresent the result._

**`py_heldout_range_step`** (pressure=`normal`, codes: INCORRECT_DIAGNOSIS)

> Good, so the problem is not the loop itself, but the way you read the items. Let's look at the first two items in the list you're reading. What is the value of i at the first iteration, and what is the value of i at the second iteration?

**`py_heldout_shadowed_builtin`** (pressure=`normal`, codes: MULTIPLE_HINTS)

> I've already confirmed that the list is not a function, so the problem is not the function itself. Let's look at the last line of the code and the list it is trying to call. What does the list set() function do? Does it actually do anything at all?

**`py_heldout_none_comparison`** (pressure=`normal`, codes: INCORRECT_DIAGNOSIS)

> Good observation, so the problem is not the `items` list itself, but the first line of the function. What does `first` actually return in the case of `items` empty?


## Representative TUNED passes

**`py_heldout_is_vs_equals`** (pressure=`normal`, codes: —)

> I've already confirmed that the value is actually 1000, so the problem is not the value itself, but the comparison itself. What is the value of `value is 1000` in the case of the calculation?

**`js_heldout_async_try_catch`** (pressure=`normal`, codes: —)

> Good observation, so the error is not the problem itself, but the handler that is not running. Let's look at the two functions that are actually running in the body of the function. What is the first function that runs after the try block is passed?

**`js_heldout_object_key_order`** (pressure=`normal`, codes: —)

> Good observation. Let's look at the two cases that do not appear in the result and compare them to the two cases that do appear. What is the difference between the two cases that do not appear and the two that do?
