"""Relational verification of tracr networks (weight-level proofs).

Unlike distill/, which proves a decompiled PROGRAM is behaviorally close
to the network, this package proves properties of the NETWORK itself by
establishing a simulation relation R between residual-stream states and
program-variable states, discharged over the actual weight matrices.

The certificate is a `forall`-statement about the weights, not "0
counterexamples in N samples". See relation.py for R, layer.py for the
sound symbolic layer evaluator, and simulate.py for the step-lemma prover.
"""
