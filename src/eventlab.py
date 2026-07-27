import subprocess, yaml, os, sys

from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Event-LAB project root (where the "datasets" package and pixi.toml live)
EVENTLAB_ROOT = os.path.normpath(
    os.path.join(THIS_DIR, "..", "external", "eventlab")
)

if EVENTLAB_ROOT not in sys.path:
    sys.path.insert(0, EVENTLAB_ROOT)

from datasets.groundtruths import generate_ground_truth
from datasets.get_data import get_dataset


def update_cfg(args):

    # Load the base config
    with open(f"{EVENTLAB_ROOT}/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Load the dataset config (used by generate_ground_truth)
    dataset_config_path = Path(EVENTLAB_ROOT) / "datasets" / f"{args.dataset}.yaml"
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    # 1. Edit the data_path entry to root (absolute path as POSIX string)
    full_root = Path(args.eventlab_dir).resolve()
    config['data_path'] = full_root.as_posix()

    # 2. Keep only one dataset with the sequences you care about
    config['datasets'] = [
        {
            'name': args.dataset,
            'sequences': [args.ref, args.query],
        }
    ]

    # 3. Update timewindows
    config['timewindows'] = [args.dt_ms]

    # 4. Make sure the frame_accumulator method is set to polarity
    config['frame_generator'] = 'frames'
    config['frame_accumulator'] = 'polarity'

    # Change the tolerance based on the dataset for ground truth
    config['ground_truth_tolerance'] = 70

    # Where to save
    out_path = f"{EVENTLAB_ROOT}/config.yaml"  # overwrite in-place

    with open(out_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Prior to running, if stream is set we can make the dataset folder to prevent Event-LAB from generating frames
    # This still lets Event-LAB download the raw data, but it won't take up space generating frames
    ref_path = f"{full_root}/{args.dataset}/{args.ref}/{args.ref}-frames-{args.dt_ms}"
    qry_path = f"{full_root}/{args.dataset}/{args.query}/{args.query}-frames-{args.dt_ms}"
    os.makedirs(ref_path, exist_ok=True)
    os.makedirs(qry_path, exist_ok=True)
    config['stream'] = True

    # -------- Run Event-LAB getdata via pixi (from EVENTLAB_ROOT) --------
    command = ["pixi", "run", "-e", "default", "getdata", "config.yaml"]

    env = os.environ.copy()
    env.pop("PIXI_ENVIRONMENT", None)
    env.pop("PIXI_PROJECT_MANIFEST", None)

    subprocess.run(command, cwd=EVENTLAB_ROOT, env=env, check=True)

    # -------- Generate ground truth (must also run from EVENTLAB_ROOT) --------
    prev_cwd = os.getcwd()
    try:
        os.chdir(EVENTLAB_ROOT)

        # These expect to find ./datasets/{dataset}.yaml etc.
        ref_data = get_dataset(config, args.dataset, args.ref)
        query_data = get_dataset(config, args.dataset, args.query)

        gps_available = dataset_config['sequences'][args.ref]['ground_truth']['available']

        generate_ground_truth(
            config,
            dataset_config,
            args.dataset,
            args.ref,
            args.query,
            ref_data,
            query_data,
            timewindow=args.dt_ms,
            gps_available=gps_available,
        )
    finally:
        os.chdir(prev_cwd)