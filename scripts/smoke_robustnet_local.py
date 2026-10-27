"""One real Cityscapes training step through the official model; never a benchmark."""

import argparse
import json
from pathlib import Path

import torch

from dgseg.data import ZipSegDataset, source_samples
from scripts.evaluate_robustnet import make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", choices=["baseline", "isw"], required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this legacy official model smoke test")
    torch.manual_seed(7)
    source = source_samples(args.data_root, "r19")["source_train"]
    image, mask, sample_id = ZipSegDataset(args.data_root, source[:1], (96, 128), train=False)[0]
    model = make_model(args.vendor, args.method).cuda().train()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    model.criterion = criterion
    model.criterion_aux = criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # The official ASPP uses BN on a 1x1 pooled feature, requiring at least two samples.
    losses = model(image.unsqueeze(0).repeat(2, 1, 1, 1).cuda(), gts=mask.unsqueeze(0).repeat(2, 1, 1).cuda(),
                   aux_gts=torch.tensor([0]), apply_wtloss=False)
    loss = losses[0] + 0.4 * losses[1]
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")
    loss.backward()
    optimizer.step()
    print(json.dumps({"method": args.method, "sample_id": sample_id, "loss": float(loss.detach().cpu()),
                      "training_steps": 1, "smoke_only": True, "publishable": False}))


if __name__ == "__main__":
    main()
