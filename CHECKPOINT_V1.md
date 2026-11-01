# Checkpoint V1 — EPI-CAVIA

## Environment
CT-graph depth 2
2 sequential decisions
4 leaf nodes
16 total states

## Methods
CAVIA
CAVIA parameter-matched
NM-CAVIA
EPI-CAVIA

## Parameters
CAVIA: 45,403
CAVIA-Matched: 48,883
EPI-CAVIA: 49,051
NM-CAVIA: 50,795

## EPI mechanism
Persistent regulatory state e derived from the adapted CAVIA context.

e' = (1-lambda)e + lambda*tanh(W_e*c' + b_e)

Hidden activity is regulated by:

h = ReLU((Wx+b) * (1 + beta*tanh(W_g*e+b_g)))

Experimental settings:
epi_size = 8
epi_decay = 0.2
epi_strength = 1.0

## Experiment
Seeds:
42
123
456
789
1001

Training:
500 meta-training iterations per seed.

## Five-seed results

Method | Whole-500 mean | Last-50 mean | Last-50 adaptation gain
CAVIA | 0.406 +/- 0.120 | 0.395 +/- 0.126 | 0.143 +/- 0.132
CAVIA-Matched | 0.484 +/- 0.083 | 0.517 +/- 0.062 | 0.264 +/- 0.058
EPI-CAVIA | 0.516 +/- 0.014 | 0.507 +/- 0.070 | 0.261 +/- 0.065
NM-CAVIA | 0.577 +/- 0.029 | 0.629 +/- 0.030 | 0.381 +/- 0.029

## Current interpretation
NM-CAVIA is strongest on the stationary CT-graph depth-2 benchmark.

EPI-CAVIA clearly improves over standard CAVIA, but is approximately equivalent
to the parameter-matched CAVIA baseline in late-training performance.

EPI-CAVIA shows particularly low between-seed variability over the complete
learning trajectory.

Therefore V1 does not establish that the epigenetic mechanism itself is superior
to additional network capacity.

## Important pending checks
Audit neutral initialization of EPI gates.
Verify exact num-test-steps evaluation semantics.
Retain parameter-matched controls.
Do not make publication-level superiority claims from V1 alone.

## Next phases
V2: CAVIA + NM + EPI combined.
V3: methylation-, histone-, chromatin-, and GRN-inspired regulation.
V4: evolutionary optimization of the regulatory mechanisms.
V5: non-stationary A -> B -> A experiments testing regulatory memory.
