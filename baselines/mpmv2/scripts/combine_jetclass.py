import argparse

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert JetClass files to a more usable format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/srv/fast/share/rodem/JetClassH5/",
        help="The path to the JetClass files",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["ZJetsToNuNu*.h5"],
        help="Glob patterns for selecting input files within each subset directory",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix appended before .h5 in the output filename "
        "(e.g. 'QCDTTbar' produces '{subset}_combined_QCDTTbar.h5')",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Which top-level splits to process (e.g. 'train' 'val'). "
        "Defaults to all splits found in data_path.",
    )
    return parser.parse_args()


def main() -> None:
    """Combine all jetclass files into a single HDF5 file."""
    # Get the arguments
    args = get_args()

    # Get the top level folders (train, val, test)
    subsets = [x for x in Path(args.data_path).iterdir() if x.is_dir()]
    print(f"Found subsets: {[x.name for x in subsets]}")
    # Cycle through each subset
    for subset in subsets:
        if args.splits is not None and subset.name not in args.splits:
            print(f"Skipping {subset.name} since it's not in the specified splits {args.splits}")
            continue
        # # Skip the train set
        # if "train" in subset.name:
        #     continue

        print(f"Processing {subset.name}")

        # Create the target file
        suffix = f"_{args.suffix}" if args.suffix else ""
        target_file = Path(args.data_path) / f"{subset.name}_combined{suffix}.h5"
        h5fw = h5py.File(target_file, mode="w")
        row = 0  # Counter for current location

        # Get a list of all files matching any of the given patterns
        files = []
        for pattern in args.patterns:
            files.extend(subset.glob(pattern))
        print(f"Found {len(files)} files matching patterns {args.patterns} in subset {subset.name}")

        # Get the name of the keys from the first file
        with h5py.File(files[0], "r") as h5fr:
            buffer = {k: [] for k in h5fr}

        # Get a list of common numbers in the file names
        # This way we ensure each buffer has one file of each type
        common_nums = np.unique([int(x.stem.split("_")[-1]) for x in files])
        for num in tqdm(common_nums):
            # Reset the buffer
            for k in buffer:
                buffer[k] = []

            # Cycle through each file
            sublist = [x for x in files if int(x.stem.split("_")[-1]) == num]
            for h5name in tqdm(sublist, leave=False):
                with h5py.File(h5name, "r") as h5fr:
                    for k in buffer:  # noqa: PLC0206
                        buffer[k].append(h5fr[k][:])

            # Shuffle each list in the buffer
            len_buff = sum(len(v) for v in buffer[k])
            order = np.random.default_rng().permutation(len_buff)
            for k in buffer:
                buffer[k] = np.concatenate(buffer[k], axis=0)[order]

            # Write the buffer to the target file
            for k, v in tqdm(buffer.items(), leave=False):
                # Create the dataset if it doesn't exist
                if row == 0:
                    h5fw.create_dataset(
                        k,
                        dtype=v.dtype,
                        shape=v.shape,
                        chunks=(1000, *v.shape[1:]),
                        maxshape=(None, *v.shape[1:]),
                    )

                # Resize the target table if it is too small
                if row + len_buff > len(h5fw[k]):
                    h5fw[k].resize((row + len_buff, *v.shape[1:]))

                # Save the data
                h5fw[k][row : row + len_buff] = v
            row += len_buff

        # Close the file
        h5fw.close()


if __name__ == "__main__":
    main()
