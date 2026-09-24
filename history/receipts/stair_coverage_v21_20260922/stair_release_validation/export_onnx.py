"""CPU conversion/parity only. Successful export is not behavior acceptance."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch


def main():
    source, dest = map(Path, sys.argv[1:3])
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(34823)
    # Export and parity are inference-only. Disable autograd before loading the
    # saved traced module so the JIT cannot cache a differentiable execution plan
    # during ONNX conversion and later mix it with inference-only tensors.
    torch.set_grad_enabled(False)
    extras = {'actuator_contract.json': '', 'training_context.json': ''}
    model = torch.jit.load(str(source), map_location='cpu', _extra_files=extras).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inputs = (torch.zeros(1, 530), torch.zeros(1, 53))
    output = dest/'thunder_cts_moe_34823.onnx'
    torch.onnx.export(model, inputs, str(output), input_names=['history', 'single_obs'],
                      output_names=['actions'], opset_version=17,
                      dynamic_axes={'history': {0: 'batch'}, 'single_obs': {0: 'batch'},
                                    'actions': {0: 'batch'}}, dynamo=False)
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=opts,
                                   providers=['CPUExecutionProvider'])
    peaks = []
    for batch in (1, 4, 72):
        for scale in (0., .1, 1., 3.):
            history = torch.randn(batch, 530)*scale
            single = torch.randn(batch, 53)*scale
            with torch.no_grad():
                expected = model(history, single).numpy()
            actual = session.run(None, {'history': history.numpy(), 'single_obs': single.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            peaks.append(float(np.max(np.abs(actual-expected))))
    for name, value in extras.items():
        if not value:
            raise RuntimeError('Missing embedded export contract: '+name)
        (dest/name).write_bytes(value)
    report = dict(conversion_passed=True, behavior_accepted=False, optimizer_updates=0,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  batches=[1, 4, 72], scales=[0., .1, 1., 3.],
                  max_absolute_error=max(peaks), atol=1e-4, rtol=1e-4,
                  onnx_version=onnx.__version__, onnxruntime_version=ort.__version__,
                  input_names=['history', 'single_obs'], input_shapes=['N,530', 'N,53'],
                  output_shape='N,16', opset=17)
    (dest/'CONVERSION.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
