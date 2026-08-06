"""Compile every RASP program in `programs/` to a transformer with TRACR.

For each program this script writes, under `compiled/<name>/`:

  params.npz         — the transformer weights (haiku params, flattened to
                       "module||param" keys; exact values, float64/32 as
                       compiled)
  model.pkl          — pickled dict {params, config, residual_labels,
                       vocab, max_seq_len, bos, pad} for exact reload
  architecture.json  — layer counts, head counts, dimensions, the meaning
                       of every residual-stream basis direction, and
                       per-parameter shapes (human/tool-readable, for
                       verification frontends)
  validation.json    — per-example comparison of the compiled model's
                       decoded output against the RASP interpreter's
                       output (ground-truth semantics)

Run:  .venv/bin/python compile_all.py [program_name ...]
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tracr.compiler import basis_inference, compiling, nodes, rasp_to_graph
from tracr.rasp import rasp

import programs

ROOT = Path(__file__).parent
COMPILED_DIR = ROOT / "compiled"
BOS = "bos"
PAD = "pad"


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def values_match(expected, actual):
    """Compare one RASP interpreter value with one decoded model value."""
    if expected is None:
        return True  # RASP 'None' output: model value is unconstrained.
    # The interpreter may return bools where the numerical model decodes
    # 0.0/1.0 (e.g. an average over a single boolean); treat as numeric.
    if isinstance(expected, bool) and is_number(actual):
        expected = float(expected)
    if is_number(expected) and is_number(actual):
        return abs(float(expected) - float(actual)) < 1e-4
    return expected == actual


def validate(module, program, model):
    """Run every example through the RASP interpreter and the compiled
    transformer; the interpreter output is the ground truth."""
    results = []
    for tokens in module.EXAMPLES:
        expected = program(tokens)
        decoded = model.apply([BOS] + tokens).decoded
        actual = decoded[1:]  # position 0 is BOS; its output is unspecified
        match = len(expected) == len(actual) and all(
            values_match(e, a) for e, a in zip(expected, actual)
        )
        results.append(
            {
                "input": tokens,
                "rasp_expected": [_jsonable(v) for v in expected],
                "model_decoded": [_jsonable(v) for v in actual],
                "match": match,
            }
        )
    return results


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def save_model(outdir, module, model):
    params = {
        f"{mod}||{name}": np.asarray(value)
        for mod, inner in model.params.items()
        for name, value in inner.items()
    }
    np.savez(outdir / "params.npz", **params)

    cfg = model.model_config
    architecture = {
        "name": module.NAME,
        "description": module.DESCRIPTION,
        "vocab": sorted(module.VOCAB, key=str),
        "max_seq_len": module.MAX_SEQ_LEN,
        "bos_token": BOS,
        "pad_token": PAD,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "key_size": cfg.key_size,
        "mlp_hidden_size": cfg.mlp_hidden_size,
        "activation_function": cfg.activation_function.__name__,
        "layer_norm": cfg.layer_norm,
        "causal": cfg.causal,
        "residual_stream_size": len(model.residual_labels),
        "residual_labels": model.residual_labels,
        "num_parameters": int(sum(p.size for p in params.values())),
        "param_shapes": {k: list(v.shape) for k, v in sorted(params.items())},
    }
    (outdir / "architecture.json").write_text(
        json.dumps(architecture, indent=2)
    )

    with open(outdir / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "params": model.params,
                "config": {
                    "num_heads": cfg.num_heads,
                    "num_layers": cfg.num_layers,
                    "key_size": cfg.key_size,
                    "mlp_hidden_size": cfg.mlp_hidden_size,
                    "dropout_rate": cfg.dropout_rate,
                    "activation_function": cfg.activation_function.__name__,
                    "layer_norm": cfg.layer_norm,
                    "causal": cfg.causal,
                },
                "residual_labels": model.residual_labels,
                "vocab": module.VOCAB,
                "max_seq_len": module.MAX_SEQ_LEN,
                "bos": BOS,
                "pad": PAD,
            },
            f,
        )
    return architecture


def io_spec(program, model, module):
    """Everything needed to run the model standalone (numpy) end to end:
    token->id encoding, and which residual dims form the output readout."""
    extracted = rasp_to_graph.extract_rasp_graph(program)
    basis_inference.infer_bases(
        extracted.graph, extracted.sink, module.VOCAB, module.MAX_SEQ_LEN
    )
    output_labels = [str(d) for d in extracted.sink[nodes.OUTPUT_BASIS]]
    categorical = rasp.is_categorical(program)
    spec = {
        "input_encoding_map": [
            [tok, idx] for tok, idx in model.input_encoder.encoding_map.items()
        ],
        "categorical_output": categorical,
        "output_labels": output_labels,
    }
    if categorical:
        inv = {i: v for v, i in model.output_encoder.encoding_map.items()}
        spec["output_values"] = [_jsonable(inv[i]) for i in range(len(inv))]
    return spec


def compile_one(module):
    program = module.make_program()
    model = compiling.compile_rasp_to_model(
        program,
        vocab=module.VOCAB,
        max_seq_len=module.MAX_SEQ_LEN,
        compiler_bos=BOS,
        compiler_pad=PAD,
    )

    outdir = COMPILED_DIR / module.NAME
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "io_spec.json").write_text(
        json.dumps(io_spec(program, model, module), indent=2)
    )
    architecture = save_model(outdir, module, model)
    results = validate(module, program, model)
    (outdir / "validation.json").write_text(json.dumps(results, indent=2))

    ok = all(r["match"] for r in results)
    print(
        f"{module.NAME:<22} layers={architecture['num_layers']} "
        f"heads={architecture['num_heads']} d_model={architecture['residual_stream_size']:>3} "
        f"params={architecture['num_parameters']:>6}  "
        f"validation={'PASS' if ok else 'FAIL'} "
        f"({sum(r['match'] for r in results)}/{len(results)} examples)"
    )
    return ok


def main():
    only = set(sys.argv[1:])
    modules = programs.load_all()
    if only:
        modules = [m for m in modules if m.NAME in only]

    all_ok = True
    for module in modules:
        try:
            all_ok &= compile_one(module)
        except Exception as e:  # keep going; report at the end
            all_ok = False
            print(f"{module.NAME:<22} COMPILE ERROR: {type(e).__name__}: {e}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
