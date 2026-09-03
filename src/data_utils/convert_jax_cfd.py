import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def _as_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return value


def _resize_frame(frame: np.ndarray, target_x_num: int, method: str) -> np.ndarray:
    if frame.shape == (target_x_num, target_x_num):
        return frame

    if frame.shape[0] % target_x_num == 0 and frame.shape[1] % target_x_num == 0:
        x_step = frame.shape[0] // target_x_num
        y_step = frame.shape[1] // target_x_num
        if method == "stride":
            return frame[::x_step, ::y_step]
        if method == "average":
            return frame.reshape(target_x_num, x_step, target_x_num, y_step).mean(axis=(1, 3))

    tensor = torch.from_numpy(frame).float()[None, None]
    return (
        F.interpolate(tensor, size=(target_x_num, target_x_num), mode="bilinear", align_corners=True)[0, 0]
        .numpy()
        .astype(frame.dtype, copy=False)
    )


def _resize_coord(coord: np.ndarray, target_x_num: int) -> np.ndarray:
    if coord.shape[0] == target_x_num:
        return coord
    if coord.shape[0] % target_x_num == 0:
        step = coord.shape[0] // target_x_num
        return coord[::step]
    return np.linspace(float(coord[0]), float(coord[-1]), target_x_num)


def convert_trajectory(
    trajectory_path: Path,
    output_path: Path,
    target_x_num: int,
    variable: str,
    method: str,
    compression: str | None,
    compression_opts: int | None,
    chunk_frames: int,
) -> None:
    files = sorted(f for f in os.listdir(trajectory_path) if f.endswith(".nc"))
    if not files:
        raise ValueError(f"No .nc files found in {trajectory_path}")

    frames = []
    times = []
    attrs = {}
    x = None
    y = None

    for frame_idx, file_name in enumerate(files):
        data_path = trajectory_path / file_name
        with h5py.File(data_path, "r") as hf:
            frame = np.asarray(hf[variable][()])
            frames.append(_resize_frame(frame, target_x_num, method))

            dt = float(np.asarray(hf.attrs["dt"]).reshape(-1)[0])
            start_time = float(np.asarray(hf.attrs.get("start_time", 0.0)).reshape(-1)[0])
            times.append(start_time + (frame_idx + 1) * dt)

            if not attrs:
                attrs = {key: _as_scalar(value) for key, value in hf.attrs.items()}
                attrs["source_x_num"] = int(frame.shape[0])
                attrs["target_x_num"] = int(target_x_num)
                attrs["downsample_method"] = method
                attrs["source_trajectory"] = str(trajectory_path)
                x = _resize_coord(np.asarray(hf["x"][()]), target_x_num) if "x" in hf else None
                y = _resize_coord(np.asarray(hf["y"][()]), target_x_num) if "y" in hf else None

    data = np.stack(frames, axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        chunks = (min(chunk_frames, data.shape[0]), target_x_num, target_x_num)
        hf.create_dataset(
            variable, data=data, compression=compression, compression_opts=compression_opts, chunks=chunks
        )
        hf.create_dataset("t", data=np.asarray(times, dtype=np.float64))
        if x is not None:
            hf.create_dataset("x", data=x)
        if y is not None:
            hf.create_dataset("y", data=y)
        for key, value in attrs.items():
            hf.attrs[key] = value


def convert(
    input_folder: str,
    output_folder: str,
    target_x_num: int = 128,
    variable: str = "w",
    method: str = "stride",
    max_trajectories: int = -1,
    overwrite: bool = False,
) -> None:
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    trajectories = sorted(
        path for path in input_path.iterdir() if path.is_dir() and path.name.startswith("trajectory_")
    )
    if max_trajectories > 0:
        trajectories = trajectories[:max_trajectories]

    for trajectory_path in trajectories:
        target_path = output_path / f"{trajectory_path.name}.h5"
        if target_path.is_file() and not overwrite:
            print(f"Skipping existing {target_path}")
            continue

        convert_trajectory(
            trajectory_path,
            target_path,
            target_x_num,
            variable,
            method,
            compression="gzip",
            compression_opts=4,
            chunk_frames=50,
        )
        print(f"Converted source ({trajectory_path}) into target ({target_path})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate JAX-CFD trajectory NetCDF frames into HDF5 files.")
    parser.add_argument(
        "--input-folder",
        default="/data/shared/jax_cfd/forced_hyper_chain/trajectories",
        help="Folder containing trajectory_* directories.",
    )
    parser.add_argument(
        "--output-folder",
        default="/data/shared/jax_cfd/forced_hyper_chain/trajectories_128",
        help="Folder where consolidated trajectory_*.h5 files are written.",
    )
    parser.add_argument("--target-x-num", type=int, default=128)
    parser.add_argument("--variable", default="w")
    parser.add_argument("--method", choices=["stride", "average", "interpolate"], default="stride")
    parser.add_argument("--max-trajectories", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        args.input_folder,
        args.output_folder,
        args.target_x_num,
        args.variable,
        args.method,
        args.max_trajectories,
        args.overwrite,
    )
