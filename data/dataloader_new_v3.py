# -*- coding: utf-8 -*-
# file: data/dataloader_new.py
#
# MeshSequenceFolder with:
# - per-case normalize (old): (V - bbox_center(ED)) / maxabs(ED)
# - global normalize (new):   (V - bbox_center(ED)) / global_scale
#   global_scale is loaded from global_scale.npz with key "global_scale"
#
# Also returns norm_center / norm_scale for de-normalization during inference/export.

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


def faces_to_edge_index(faces, bidirectional=True):
    """
    faces: (F, 3) int64
    return edge_index: (2, E) int64 —— adapt to torch_geometric
    """
    f = faces.astype(np.int64)
    edges = set()
    tri = [(0, 1), (1, 2), (2, 0)]
    for a, b, c in f:
        tri_v = [a, b, c]
        for i, j in tri:
            u, v = tri_v[i], tri_v[j]
            if u == v:
                continue
            edges.add((u, v))
            if bidirectional:
                edges.add((v, u))
    edge_index = np.array(list(edges), dtype=np.int64).T  # (2, E)
    return edge_index


def read_table_auto(path: str) -> pd.DataFrame:
    """
    Support .csv / .xlsx / .xls / .tsv / .txt
    """
    assert os.path.isfile(path), f"Missing table file: {path}"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path} (ext={ext})")


class MeshSequenceFolder(Dataset):
    """
    root_dir/
      ├─ faces_fhm_numpy.npy
      └─ <subject_id>/ (e.g. 000,001,...)
           ├─ time00/fhm.npy
           ├─ time01/fhm.npy
           └─ ...
    """

    def __init__(self,
                 root_dir,
                 seq_len=25,
                 cond_dim=10,
                 normalize=True,

                 # Normalization
                 normalize_mode="global",      # "per_case" | "global"
                 global_scale_path=None,         # path to global_scale.npz (key: "global_scale")

                 # --- label table ---
                 excel_path=None,            # csv/xlsx/tsv
                 patient_id_col="patient_id",
                 group_col="Group",
                 label2id=None,              # e.g. {"NOR":0,"DCM":1,"HCM":2,"RV":3}
                 default_label="NOR",
                 # --- filtering ---
                 filter_groups=None,          # e.g. "HCM" or ["HCM","DCM"]
                 drop_unlabeled=False):
        self.root = root_dir
        self.seq_len = seq_len
        self.normalize = bool(normalize)
        self.normalize_mode = str(normalize_mode)
        assert self.normalize_mode in ["per_case", "global"], f"normalize_mode must be 'per_case' or 'global', got {self.normalize_mode}"
        self.cond_dim = cond_dim

        # faces & edges (global, compute once)
        faces_path = os.path.join(root_dir, "faces_fhm_numpy.npy")
        assert os.path.isfile(faces_path), f"Missing faces file: {faces_path}"
        faces = np.load(faces_path)                    # (F,3)
        self.faces = torch.from_numpy(faces).long()    # (F,3)
        edge_index = faces_to_edge_index(faces)        # (2,E)
        self.edge_index = torch.from_numpy(edge_index).long()

        # enumerate subjects
        self.subjects = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d != "__pycache__"
        ])
        self.subj_root = root_dir

        # Dataset-wide scale
        self.global_scale = None
        self._global_scale_path_used = None
        if self.normalize and self.normalize_mode == "global":
            if global_scale_path is None or str(global_scale_path).strip() == "":
                # Use the split-local scale file if no path is provided
                global_scale_path = os.path.join(root_dir, "global_scale.npz")

            self._global_scale_path_used = global_scale_path
            if os.path.isfile(global_scale_path):
                d = np.load(global_scale_path)
                if "global_scale" not in d.files:
                    raise KeyError(f"'global_scale' not found in {global_scale_path}, keys={d.files}")
                self.global_scale = float(d["global_scale"])
                if not np.isfinite(self.global_scale) or self.global_scale <= 0:
                    raise ValueError("global_scale must be positive and finite")
                print(f"[Dataset] Loaded global_scale={self.global_scale:.6f} from {global_scale_path}")
            else:
                # Fallback for data without a global-scale file
                raise FileNotFoundError(f"Global scaling file not found: {global_scale_path}")

        # ---- label mapping ----
        if label2id is None:
            label2id = {"NOR": 0, "DCM": 1, "HCM": 2, "RV": 3}
        self.label2id = {k.upper(): int(v) for k, v in label2id.items()}
        assert default_label.upper() in self.label2id, f"default_label {default_label} not in label2id"
        self.default_cid = self.label2id[default_label.upper()]

        # ---- table: patient_id -> class_id ----
        self.pid2cid = None
        if excel_path is not None and str(excel_path).strip() != "":
            df = read_table_auto(excel_path)

            # tolerate column name case/whitespace
            cols = {c.strip(): c for c in df.columns}

            def find_col(name):
                if name in cols:
                    return cols[name]
                for k in cols:
                    if k.lower() == name.lower():
                        return cols[k]
                return None

            pid_col = find_col(patient_id_col)
            grp_col = find_col(group_col)
            assert pid_col is not None, f"Cannot find patient_id_col='{patient_id_col}' in {list(df.columns)}"
            assert grp_col is not None, f"Cannot find group_col='{group_col}' in {list(df.columns)}"

            pid_list = df[pid_col].astype(int).tolist()
            grp_list = df[grp_col].astype(str).str.upper().tolist()

            pid2cid = {}
            for p, g in zip(pid_list, grp_list):
                if g not in self.label2id:
                    continue
                pid2cid[int(p)] = self.label2id[g]
            self.pid2cid = pid2cid

        # ---- filter_groups ----
        self.filter_groups = None
        if filter_groups is not None:
            if isinstance(filter_groups, str):
                filter_groups = [filter_groups]
            self.filter_groups = [g.upper() for g in filter_groups]

        if self.filter_groups is not None:
            assert self.pid2cid is not None, "filter_groups requires excel_path/table to provide patient_id->Group"
            inv = {v: k for k, v in self.label2id.items()}

            kept = []
            for sid in self.subjects:
                pid = self._sid_to_patient_id(sid)
                if pid is None:
                    continue
                cid = self.pid2cid.get(pid, None)
                if cid is None:
                    if drop_unlabeled:
                        continue
                    cid = self.default_cid
                lab = inv.get(cid, None)
                if lab is None:
                    continue
                if lab in self.filter_groups:
                    kept.append(sid)

            self.subjects = kept
            print(f"[Dataset] root={root_dir} filter_groups={self.filter_groups} -> {len(self.subjects)} subjects")

    def __len__(self):
        return len(self.subjects)

    @staticmethod
    def _sid_to_patient_id(sid: str):
        # dir is "000" -> int("000") = 0
        try:
            return int(sid)
        except Exception:
            digits = ''.join([c for c in sid if c.isdigit()])
            return int(digits) if digits != "" else None

    def _load_sequence(self, subj_dir):
        frame_dirs = sorted(glob.glob(os.path.join(subj_dir, "time*")))
        if self.seq_len is not None:
            frame_dirs = frame_dirs[:self.seq_len]

        verts_list = []
        for fd in frame_dirs:
            vpath = os.path.join(fd, "fhm.npy")
            assert os.path.isfile(vpath), f"Missing {vpath}"

            fhm = np.load(vpath, allow_pickle=True)

            # A) stored as (N,3)
            if isinstance(fhm, np.ndarray) and fhm.ndim == 2 and fhm.shape[1] == 3:
                v = fhm
            else:
                # B) dict: {'V': (N,3)}
                obj = fhm.item()
                if isinstance(obj, dict) and "V" in obj:
                    v = obj["V"]
                elif isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 3:
                    v = obj
                else:
                    raise ValueError(f"Unknown fhm.npy format: {vpath}, type={type(obj)}")

            verts_list.append(torch.from_numpy(v).float())

        V = torch.stack(verts_list, dim=0)  # (T,N,3)
        return V

    # ---------- normalize helpers ----------
    @staticmethod
    def _bbox_center(V0: torch.Tensor):
        # V0: (N,3)
        return (V0.max(dim=0).values + V0.min(dim=0).values) / 2.0

    def _normalize_per_case(self, V: torch.Tensor):
        """
        per-case normalize:
          center = bbox_center(ED)
          scale  = maxabs(ED)   (scalar)
          Vn = (V - center) / scale
        return: Vn, center(3,), scale(float)
        """
        V0 = V[0]
        center = self._bbox_center(V0)  # (3,)
        Vn = V - center[None, None, :]
        scale = float(V0.abs().max().item())
        Vn = Vn / (scale + 1e-9)
        return Vn, center, scale

    def _normalize_global(self, V: torch.Tensor):
        """
        global normalize:
          center = bbox_center(ED)   (still per-case translation removal)
          scale  = global_scale      (fixed across dataset)
          Vn = (V - center) / global_scale
        return: Vn, center(3,), scale(float==global_scale)
        """
        assert self.global_scale is not None, (
            "global_scale not loaded. Set normalize_mode='global' and provide valid global_scale_path."
        )
        V0 = V[0]
        center = self._bbox_center(V0)  # (3,)
        Vn = V - center[None, None, :]
        scale = float(self.global_scale)
        Vn = Vn / (scale + 1e-9)
        return Vn, center, scale

    def __getitem__(self, idx):
        sid = self.subjects[idx]
        subj_dir = os.path.join(self.subj_root, sid)

        V = self._load_sequence(subj_dir)  # (T,N,3)

        # default norm stats (identity)
        norm_center = torch.zeros(3, dtype=torch.float32)
        norm_scale = torch.tensor(1.0, dtype=torch.float32)

        if self.normalize:
            if self.normalize_mode == "global":
                V, center, scale = self._normalize_global(V)
            else:
                V, center, scale = self._normalize_per_case(V)

            norm_center = center.detach().cpu().float()
            norm_scale = torch.tensor(float(scale), dtype=torch.float32)

        T, N, _ = V.shape

        faces_T = self.faces.unsqueeze(0).repeat(T, 1, 1)             # (T,F,3)
        edges_T = self.edge_index.unsqueeze(0).repeat(T, 1, 1)        # (T,2,E)

        # continuous condition placeholder (if unused, cond_dim=0)
        con = torch.zeros(self.cond_dim, dtype=torch.float32)

        # cond_id from table (if available)
        pid = self._sid_to_patient_id(sid)
        if self.pid2cid is None or pid is None:
            cid = self.default_cid
        else:
            cid = self.pid2cid.get(pid, self.default_cid)
        cond_id = torch.tensor(cid, dtype=torch.long)

        return {
            "v": V,
            "f": faces_T,
            "edge_index": edges_T,
            "con": con,
            "cond_id": cond_id,           # (,) long
            "sid": sid,
            "patient_id": pid,

            # Denormalization statistics
            "norm_center": norm_center,   # (3,)
            "norm_scale": norm_scale,     # scalar
            "norm_mode": self.normalize_mode,
        }