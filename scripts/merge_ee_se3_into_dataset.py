#!/usr/bin/env python3
"""Merge the derived EE SE(3) columns into the dataset's own episode parquets.

This rewrites the source data files in place. Each file is streamed row-group
by row-group into a temporary file in the same directory, fully verified
against the original, and only then atomically renamed over it -- so an
interrupted or failed run always leaves the original intact.

The parquet's embedded `huggingface` schema metadata and meta/info.json are
both extended so the new columns are visible to the LeRobot loader.

Run compute_ee_se3_all.py first.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compute_ee_se3 as C  # noqa: E402

DATASET = C.DATASET
JOIN_COLS = {"index", "episode_index", "frame_index", "timestamp"}

# names for the LeRobot info.json feature entries
# str(pa.float32()) is "float", but the dataset declares "float32"
ARROW_DTYPE = {
    "float": "float32", "double": "float64",
    "int64": "int64", "int32": "int32", "bool": "bool",
}

MATRIX_NAMES = [f"m{r}{c}" for r in range(4) for c in range(4)]
POS_NAMES = ["x", "y", "z"]
QUAT_NAMES = ["qw", "qx", "qy", "qz"]


def ee_columns(table):
    """The derived columns to merge, in stable order (join keys excluded)."""
    return [n for n in table.column_names if n not in JOIN_COLS]


def dtype_name(arrow_type):
    key = str(arrow_type)
    if key not in ARROW_DTYPE:
        raise RuntimeError(f"unmapped arrow dtype {key!r}")
    return ARROW_DTYPE[key]


def hf_feature(field):
    """datasets-style feature spec for one of our fixed-size-list columns."""
    if pa.types.is_fixed_size_list(field.type):
        return {"feature": {"dtype": dtype_name(field.type.value_type), "_type": "Value"},
                "length": field.type.list_size, "_type": "List"}
    return {"dtype": dtype_name(field.type), "_type": "Value"}


def lerobot_feature(field):
    if pa.types.is_fixed_size_list(field.type):
        n = field.type.list_size
        names = {16: MATRIX_NAMES, 3: POS_NAMES, 4: QUAT_NAMES}.get(n, [str(i) for i in range(n)])
        return {"dtype": dtype_name(field.type.value_type), "shape": [n], "names": names}
    return {"dtype": dtype_name(field.type), "shape": [1], "names": None}


def source_codecs(pf):
    """Per-leaf compression codec of the existing file, to preserve it."""
    codecs = {}
    rg = pf.metadata.row_group(0)
    for i in range(rg.num_columns):
        col = rg.column(i)
        codec = col.compression.lower()
        # pyarrow spells the absence of compression "none", parquet says
        # "uncompressed" when reading it back
        codecs[col.path_in_schema] = "none" if codec == "uncompressed" else codec
    return codecs


def leaf_paths(field):
    """Parquet leaf path(s) for one arrow field."""
    if pa.types.is_fixed_size_list(field.type):
        return [f"{field.name}.list.element"]
    return [field.name]


def merge_file(src, derived, dry_run=False):
    pf = pq.ParquetFile(src)
    ee = pq.read_table(derived)
    names = ee_columns(ee)

    if pf.metadata.num_rows != len(ee):
        raise RuntimeError(f"row mismatch: {src.name} has {pf.metadata.num_rows}, "
                           f"{derived.name} has {len(ee)}")

    existing = set(pf.schema_arrow.names)
    overlap = existing & set(names)
    if overlap:
        if overlap == set(names):
            return None, None, pf.metadata.num_rows   # already merged; skip
        raise RuntimeError(f"{src.name} contains only SOME EE columns "
                           f"({len(overlap)}/{len(names)}) -- looks like a partial "
                           f"write; restore this file before re-running")

    # verify the derived rows line up with the source rows
    src_keys = pq.read_table(src, columns=["index", "frame_index"])
    for k in ("index", "frame_index"):
        if not np.array_equal(src_keys[k].to_numpy(), ee[k].to_numpy()):
            raise RuntimeError(f"{src.name}: {k} does not match derived file")

    # build output schema: original fields + EE fields, metadata extended
    new_fields = [ee.schema.field(n) for n in names]
    out_schema = pa.schema(list(pf.schema_arrow) + new_fields)
    meta = dict(pf.schema_arrow.metadata or {})
    hf = json.loads(meta[b"huggingface"].decode())
    for f in new_fields:
        hf["info"]["features"][f.name] = hf_feature(f)
    meta[b"huggingface"] = json.dumps(hf).encode()
    out_schema = out_schema.with_metadata(meta)

    if dry_run:
        return names, new_fields, pf.metadata.num_rows

    # preserve each original leaf's codec; the new float columns get snappy
    comp = source_codecs(pf)
    for f in new_fields:
        for leaf in leaf_paths(f):
            comp[leaf] = "snappy"
    tmp = src.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(tmp, out_schema, compression=comp,
                              use_dictionary=False)
    try:
        offset = 0
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            n = tbl.num_rows
            for f in new_fields:
                tbl = tbl.append_column(f, ee[f.name].slice(offset, n).combine_chunks())
            writer.write_table(tbl.replace_schema_metadata(meta))
            offset += n
        writer.close()
    except Exception:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise

    verify(src, tmp, ee, names)
    os.replace(tmp, src)
    return names, new_fields, offset


def verify(src, tmp, ee, names):
    """Every original column byte-identical, every new column correct."""
    a, b = pq.ParquetFile(src), pq.ParquetFile(tmp)
    if a.metadata.num_rows != b.metadata.num_rows:
        raise RuntimeError(f"{tmp.name}: row count changed")
    if a.num_row_groups != b.num_row_groups:
        raise RuntimeError(f"{tmp.name}: row group count changed")
    orig = a.schema_arrow.names
    if b.schema_arrow.names[:len(orig)] != orig:
        raise RuntimeError(f"{tmp.name}: original columns reordered or renamed")

    offset = 0
    for rg in range(a.num_row_groups):
        ta, tb = a.read_row_group(rg), b.read_row_group(rg)
        for name in orig:
            if not ta[name].combine_chunks().equals(tb[name].combine_chunks()):
                raise RuntimeError(f"{tmp.name}: column {name} differs in row group {rg}")
        for name in names:
            got = tb[name].combine_chunks()
            want = ee[name].slice(offset, ta.num_rows).combine_chunks()
            if not got.equals(want):
                raise RuntimeError(f"{tmp.name}: merged column {name} wrong in row group {rg}")
        offset += ta.num_rows


def update_info_json(new_fields, dataset):
    path = dataset / "meta/info.json"
    backup = dataset / "meta/info.json.bak"
    info = json.loads(path.read_text())
    if not backup.exists():
        shutil.copy2(path, backup)
    for f in new_fields:
        info["features"][f.name] = lerobot_feature(f)
    path.write_text(json.dumps(info, indent=4))
    return path, backup


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=pathlib.Path, default=DATASET)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    D = args.dataset
    eps = pq.read_table(D / "meta/episodes/chunk-000/file-000.parquet",
                        columns=["episode_index", "data/chunk_index",
                                 "data/file_index"]).to_pandas()
    todo = args.only if args.only is not None else eps.episode_index.tolist()

    new_fields = None
    total = 0
    for ep in todo:
        row = eps[eps.episode_index == ep].iloc[0]
        src = (D / "data" / f"chunk-{row['data/chunk_index']:03d}"
               / f"file-{row['data/file_index']:03d}.parquet")
        derived = D / "derived" / f"ee_se3_episode_{ep:03d}.parquet"
        if not derived.exists():
            raise SystemExit(f"missing {derived}; run compute_ee_se3_all.py first")
        before = src.stat().st_size
        names, fields, n = merge_file(src, derived, args.dry_run)
        if names is None:
            print(f"ep {ep:3d}  skip (already merged)  {src.name}")
            continue
        new_fields = fields
        after = src.stat().st_size
        total += n
        verb = "would add" if args.dry_run else "added"
        print(f"ep {ep:3d}  {verb} {len(names)} cols to {src.name}  "
              f"{n} rows  {before/1e6:.0f} -> {after/1e6:.0f} MB")

    if not args.dry_run and new_fields:
        path, backup = update_info_json(new_fields, D)
        print(f"\nupdated {path}  (original saved as {backup.name})")
    print(f"done: {len(todo)} files, {total} rows")


if __name__ == "__main__":
    raise SystemExit(main())
