import argparse
import pyloudnorm as pyln
import os
import soundfile as sf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", )


# FIXED_LOUDNESS = -23.0


def make_dataset(dir):
    files = []
    assert os.path.isdir(dir) or os.path.islink(
        dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            path = os.path.join(root, fname)
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True, help="Location of the files")
    parser.add_argument("--output", type=str, required=True, help="loaction of londnorm audio")
    parser.add_argument("--fixed_loudness", type=float, default=-23.0, help="target loudness")
    opt = parser.parse_args()

    files_loc = opt.dataroot
    files = make_dataset(files_loc)

    results = Path(opt.output)
    results.mkdir(parents=True, exist_ok=True)
    # try:
    #    os.mkdir(results)
    # except OSError:
    #    print("Error in creating RESULTS directory")

    for f in files:
        file_name = os.path.split(f)[-1]
        data, sr = sf.read(f)
        # offical default set,
        # meter = pyln.Meter(sr)  # create BS.1770 meter,400ms block size
        meter = pyln.Meter(sr, block_size=0.200)

        loudness = meter.integrated_loudness(data)
        data_normalized = pyln.normalize.loudness(data, loudness, target_loudness=opt.fixed_loudness)
        sf.write(os.path.join(results, file_name), data_normalized, sr)
        # print("Saved: %s" % (file_name))
        logging.info(f"Saved: {file_name}")

    return


if __name__ == "__main__":
    main()
