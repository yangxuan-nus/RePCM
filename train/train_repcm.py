"""Train the RePCM completion model with full sequences and an ED-conditioned prior."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from data.dataloader_new_v3 import MeshSequenceFolder
from loss.mesh_loss import MSE_loss
from model.repcm import RePCM


def parse_args():
    parser = argparse.ArgumentParser(description="Train RePCM")
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--template_dir", required=True, type=Path)
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--global_scale_path", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--proto_momentum", type=float, default=0.95)
    parser.add_argument("--num_val_samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_region_assets(template_dir, n_vertices, device):
    assets = {
        "vertex_token_id": template_dir / "region_id_K16.npy",
        "idx_fixed": template_dir / "idx_fixed_regionwise512.npy",
        "A_allow": template_dir / "A16_bool.npy",
        "AK_allow": template_dir / "A512_allow_fromA16.npy",
    }
    for path in assets.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing motion-based region asset: {path}")

    region_id = torch.from_numpy(np.load(assets["vertex_token_id"]).astype(np.int64)).to(device)
    anchors = torch.from_numpy(np.load(assets["idx_fixed"]).astype(np.int64)).to(device)
    region_adj = torch.from_numpy(np.load(assets["A_allow"]).astype(bool)).to(device)
    anchor_adj = torch.from_numpy(np.load(assets["AK_allow"]).astype(bool)).to(device)

    if region_id.shape != (n_vertices,):
        raise ValueError(f"Expected {n_vertices} vertex region labels, got {tuple(region_id.shape)}")
    if anchors.shape != (512,) or anchors.min() < 0 or anchors.max() >= n_vertices:
        raise ValueError("Expected 512 valid anchor indices")
    if region_adj.shape != (16, 16) or anchor_adj.shape != (512, 512):
        raise ValueError("Expected adjacency matrices of size 16x16 and 512x512")
    return region_id, anchors, region_adj, anchor_adj


def build_model(seq_len, n_vertices, template_dir, device):
    model = RePCM(
        T=seq_len,
        latent_dim=16,
        num_traj=512,
        num_tokens=16,
        num_experts=4,
        enc_depth=8,
        dec_depth=8,
    ).to(device)
    region_id, anchors, region_adj, anchor_adj = load_region_assets(template_dir, n_vertices, device)
    model.set_fixed_idx(anchors)
    model.set_vertex_token_id(region_id)
    model.build_anchor_token_id()
    model.set_token_graph(region_adj)
    model.set_anchor_graph(anchor_adj)
    return model


def build_loader(root, batch, workers, scale_path, shuffle):
    ds = MeshSequenceFolder(
        root_dir=str(root),
        seq_len=25,
        cond_dim=0,
        normalize=True,
        normalize_mode="global",
        global_scale_path=str(scale_path),
    )
    if len(ds) == 0:
        raise ValueError(f"Empty dataset: {root}")
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True)


def train_epoch(model, loader, optimizer, device, beta, prototype_momentum):
    model.train()
    sum_loss, count = 0.0, 0
    for batch in loader:
        sequence = batch["v"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(pc=sequence, queries=sequence[:, 0], single_frame=False)
        rec = MSE_loss(output["recon_pc"], sequence)
        kl = output["kl_temp"].mean()
        loss = rec + beta * kl
        loss.backward()
        optimizer.step()
        model.update_shape_prototypes(output["e_s"], momentum=prototype_momentum)
        bs = sequence.size(0)
        sum_loss += float(loss.detach()) * bs
        count += bs
    return sum_loss / count


@torch.no_grad()
def validate(model, loader, device, num_samples):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        sequence = batch["v"].to(device)
        ed = sequence[:, :1]
        pred = None
        for _ in range(num_samples):
            sample = model(pc=ed, queries=ed[:, 0], single_frame=True, return_extra=False)["recon_pc"]
            pred = sample if pred is None else pred + sample
        pred = pred / num_samples
        loss = MSE_loss(pred, sequence)
        bs = sequence.size(0)
        total += float(loss) * bs
        count += bs
    return total / count


def save_checkpoint(path, model, optimizer, epoch, best_loss):
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": best_loss,
    }, path)


def main():
    args = parse_args()
    if args.batch < 1 or args.num_val_samples < 1:
        raise ValueError("Batch size and validation sample count must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    scale_path = args.global_scale_path or (args.data_root / "global_scale.npz")
    train_loader = build_loader(args.data_root / "train", args.batch, args.num_workers, scale_path, True)
    val_loader = build_loader(args.data_root / "val", args.batch, args.num_workers, scale_path, False)
    n_vertices = train_loader.dataset[0]["v"].shape[1]
    model = build_model(25, n_vertices, args.template_dir, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch, best_loss, stale = 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("val_loss", best_loss))

    for epoch in trange(start_epoch, args.n_epochs, desc="RePCM"):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.beta, args.proto_momentum)
        val_loss = validate(model, val_loader, device, args.num_val_samples)
        print(f"epoch {epoch + 1}: train={train_loss:.6f}, ED-only val={val_loss:.6f}")
        save_checkpoint(args.model_dir / "last.pt", model, optimizer, epoch, val_loss)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            save_checkpoint(args.model_dir / "best.pt", model, optimizer, epoch, val_loss)
        else:
            stale += 1
            if stale >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


if __name__ == "__main__":
    main()
