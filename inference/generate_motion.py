"""Generate full-cycle biventricular motion from a single ED mesh."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataloader_new_v3 import MeshSequenceFolder
from train.train_repcm import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="RePCM ED-only inference")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--template_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--global_scale_path", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_vtp", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def synthesize_sequence(model, ed, num_samples=15):
    """Return the mean ED-conditioned sample in normalized coordinates."""
    if ed.ndim != 4 or ed.shape[1] != 1 or ed.shape[-1] != 3:
        raise ValueError("Expected an ED mesh shaped (B,1,N,3)")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    result = None
    for _ in range(num_samples):
        sample = model(pc=ed, queries=ed[:, 0], single_frame=True,
                       return_extra=False)["recon_pc"]
        result = sample if result is None else result + sample
    return result / num_samples


def save_vtp_sequence(path, vertices, faces):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("Install pyvista for VTP export") from exc
    faces = faces.astype(np.int64)
    cells = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    for t, frame in enumerate(vertices):
        pv.PolyData(frame, cells).save(str(path / f"t{t:03}.vtp"))


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    scale_path = args.global_scale_path or (args.data_root / "global_scale.npz")
    dataset = MeshSequenceFolder(
        root_dir=str(args.data_root / args.split),
        seq_len=25,
        cond_dim=0,
        normalize=True,
        normalize_mode="global",
        global_scale_path=str(scale_path),
    )
    if len(dataset) == 0:
        raise ValueError("Empty inference dataset")

    n_vertices = dataset[0]["v"].shape[1]
    model = build_model(25, n_vertices, args.template_dir, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    latent_dim = int(state_dict["mean_fc_x0.weight"].shape[0])
    if latent_dim != model.latent_dim:
        raise ValueError(f"Checkpoint latent dim {latent_dim} does not match {model.latent_dim}")
    model.load_state_dict(state_dict, strict=True)
    if not bool(model.shape_protos_valid.item()):
        raise RuntimeError("Checkpoint is missing fitted EMA shape prototypes")
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="RePCM inference"):
        sample = dataset[idx]
        vertices = sample["v"].unsqueeze(0).to(device)
        ed = vertices[:, :1]
        prediction = synthesize_sequence(model, ed, args.num_samples)[0].cpu().numpy()
        target = vertices[0].cpu().numpy()
        center = sample["norm_center"].cpu().numpy()
        scale = float(sample["norm_scale"])
        prediction = prediction * scale + center[None, None, :]
        target = target * scale + center[None, None, :]
        faces = sample["f"][0].cpu().numpy()
        subject_dir = args.out_dir / str(sample["sid"])
        subject_dir.mkdir(parents=True, exist_ok=True)
        np.save(subject_dir / "pred.npy", prediction.astype(np.float32))
        np.save(subject_dir / "gt.npy", target.astype(np.float32))
        np.save(subject_dir / "faces.npy", faces.astype(np.int64))
        if args.save_vtp:
            pred_dir = subject_dir / "pred_vtp"
            pred_dir.mkdir(exist_ok=True)
            save_vtp_sequence(pred_dir, prediction, faces)


if __name__ == "__main__":
    main()
