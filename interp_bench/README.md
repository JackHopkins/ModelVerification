---
license: cc-by-4.0
---

# InterpBench

This repository of models is complimentary to [InterpBench's code repository](https://github.com/FlyingPumba/InterpBench), and should be used to load the models. 

An example on how to use them can be found in this [DEMO notebook](https://github.com/FlyingPumba/InterpBench/blob/main/DEMO_InterpBench.ipynb).
Alternatively, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) can also be used to load it using the ll_config.json

**Warning**: Using InterpBench models in this repo as ground truth circuits for evaluating circuit discovery techniques requires extra considerations on the granularity of the comparison to be sound.
Most techniques work at the QKV granularity level, and thus they consider the outputs of the Q, K, and V matrices in attention heads and the output of MLP components as nodes in the computational graph. On the other hand, InterpBench models are trained at the attention head level, without putting a constraint on the head subcomponents, which means that the trained models can solve the required tasks via QK circuits, OV circuits, or a combination of both. Thus, during the evaluation of circuit discovery techniques, QKV nodes need to be promoted to heads on the discovered circuits. In other words, if for example, the output of a Q matrix in an attention head is deemed as part of the circuit, you should also consider the whole attention head to be part of it as well.

## Structure

Each directory corresponds to a model/datapoint in the InterpBench dataset. It is structured as: 

```
- task // directory name
-- ll_model.pth // the low level transformer model
-- ll_model_cfg.pkl // a config for the transformer model
-- meta.json // training hyperparams
-- edges.pkl // label for the circuit, i.e., list of all the edges that are a part of the ground truth circuit 
```

## Paper

The full paper can be read in arXiv: [InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques](https://arxiv.org/abs/2407.14494).

For citing, please use:

```
@misc{gupta2024interpbenchsemisynthetictransformersevaluating,
      title={InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques}, 
      author={Rohan Gupta and Iván Arcuschin and Thomas Kwa and Adrià Garriga-Alonso},
      year={2024},
      eprint={2407.14494},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14494}, 
}
```