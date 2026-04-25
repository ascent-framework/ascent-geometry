# H1a/H1b 3-Task Preview — 2026-04-25

## Scope

- **Exploratory preview** — not registered evidence (N=3, minimum=10)
- Input: CommonsenseQA, ARC-Challenge, HellaSwag (1000-step each)
- Model: Qwen/Qwen2.5-1.5B-Instruct

## Results

### H1a (inconclusive — N<10)
- r90: 3 / 3
- rho: 1.0
- Bootstrap 95% CI: [0.3333333333333333, 1.0]
- Bootstrap r90 mean: 2.125
- Singular values: [1.000206, 1.000022, 0.999777]

### H1b
- Mean |cos|: 0.000369
- Max |cos|: 0.000775
- Decision: **pass**

## Key Observation vs 50-step Pilot

| Metric | 50-step (10 tasks) | 1000-step (3 tasks) |
|--------|:-----------------:|:-------------------:|
| H1b mean |cos| | 0.086 | **0.00037** |
| Singular values | ~1.0 uniform | ~1.0 uniform |

H1b mean |cos| dropped 200x from 50-step pilot to 1000-step.
Task vectors are now pointing in highly distinct directions,
confirming that 1000-step training produces task-specific signals
rather than initialization-noise dominated vectors.

## Prediction for 10-task Full Set

- **H1b: likely pass** — directional diversity already clear at N=3
- **H1a: uncertain** — shared subspace detection requires all 10 tasks,
  especially the math cluster (GSM8K, MATH, AIME, AMC, MATH500) vs
  coding (HumanEval, MBPP) vs commonsense (CommonsenseQA, HellaSwag, ARC-Challenge)

## Vector Norms
- CommonsenseQA: 23.1648
- ARC-Challenge: 23.1703
- HellaSwag: 23.1067

(50-step pilot norms were ~22.88–22.89)
