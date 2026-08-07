import argparse, os, time

from loguru import logger
from tqdm import tqdm
from src.inference import run
from src.imagevpr import run as run_images
from src.traversevpr import run as run_traverse_npz
from src.eventlab import update_cfg

# Datasets that are folders of independent images rather than continuous recordings. They
# take a different path end to end: no Event-LAB download or dt_ms slicing, and positives
# come from per-image geography or the dataset's supplied ground truth.
IMAGE_DATASETS = {"msls", "nycevent", "pitts", "pitts250k", "tokyo247"}
# Dataset-native split names used when --ref/--query are left unset.
IMAGE_SPLITS = {
    "msls": ("database", "query"),
    "nycevent": ("database", "queries"),
    # `pitts` is same-panorama view retrieval off a supplied ground truth, NOT place
    # recognition: its positives are the other 23 tiles of the query's own panorama.
    # `pitts250k` is the official NetVLAD test split, where database and query tiles come
    # from different Street View captures and positives are a 25 m GPS band.
    "pitts": ("ref", "query"),
    "pitts250k": ("database", "queries"),
    "tokyo247": ("database", "queries"),
}

def eventlab(args):
    # Check if dataset exists, if not download it
    if not os.path.exists(f"{args.eventlab_dir}/{args.dataset}/{args.ref}") or not os.path.exists(f"{args.eventlab_dir}/{args.dataset}/{args.query}") or not os.path.exists(f"{args.eventlab_dir}/{args.dataset}/ground_truth/{args.ref}_{args.query}_GT.npy"):
        update_cfg(args)
    else:
        logger.info(f"Data exists at {args.eventlab_dir}/{args.dataset}")

def inference(args):
    if args.dataset in IMAGE_DATASETS:
        logger.info(f"Running {args.method} on {args.dataset}: DB {args.ref} <--> "
                    f"Q {args.query}, {args.positive_dist_threshold:g} m ground-truth radius.")
        run_images(args)
    elif args.source:
        logger.info(f"Running {args.method} on {args.dataset} [{args.source} events]: "
                    f"DB {args.ref} <--> Q {args.query} @ {args.dt_ms} ms.")
        run_traverse_npz(args)
    else:
        logger.info(f"Running inference on {args.dataset}: DB {args.ref} <--> Q {args.query} @ {args.dt_ms} ms.")
        run(args)

def main():

    parser = argparse.ArgumentParser(description="Args for megaevent.")

    # Inference parameters
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["brisbane_event", "msls", "nsavp", "nycevent", "pitts", "pitts250k", "tokyo247"],
                        help="Sets the dataset to be used for inference.")
    parser.add_argument("--ref", "-r", type=str, default=None,
                        help="Sets the reference dataset for inference. Image datasets "
                             "use their standard database/reference split by default.")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Sets the query dataset for inference. Image datasets "
                             "use their standard query split by default.")
    parser.add_argument("--dt-ms", type=int, default=50,
                        help="Sets the time window in milliseconds for inference.")
    parser.add_argument("--model", "-m", type=str, default="s_salad_ft4", choices=["s_salad_ft4", "s_gem_ft4"],
                        help="Sets the model to be used for inference.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Explicit MegaEvent checkpoint. Required for nycevent so a "
                             "result never depends on a changing model-name default.")
    parser.add_argument("--feature-dir", type=str, default="./features",
                        help="Sets the directory to save features.")
    # Event-LAB args
    parser.add_argument("--eventlab-dir", type=str, default="./data",
                        help="Sets the Event-LAB default dataset directory.")
    # Image-set args (tokyo247 and friends)
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Root of the image-set datasets: <data-dir>/<dataset>/numpy/<split>.")
    parser.add_argument("--positive-dist-threshold", type=float, default=25.0,
                        help="Metres within which a retrieved database image counts as correct.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: roughly this many frames per side. Image sets "
                             "take the database images nearest a query; traverses take a "
                             "uniform stride over both, which keeps the ground-truth band "
                             "diagonal. Caches and artifacts are named separately, and the "
                             "recall is not comparable to a full run either way.")
    # Real-vs-synthetic ablation: score a traverse from materialised .npz frames instead of
    # its own event stream, so the same traverse can be run twice from two event sources.
    parser.add_argument("--source", type=str, default=None,
                        choices=["real", "i2e", "real_masked", "i2e_masked"],
                        help="Score a traverse from pre-built .npz frames rather than the "
                             "HDF5 stream: 'real' is the recording's own event slices, "
                             "'i2e' is an I2E micro-saccade over the DAVIS APS frame from "
                             "the same instant. The '_masked' variants drop the dead "
                             "vignette, where I2E's log transform turns read noise into "
                             "most of its events. Setting it also enables --method for a "
                             "traverse dataset. Build them with scripts/dump_event_npz.py, "
                             "scripts/extract_aps.py and scripts/mask_vignette.py.")
    parser.add_argument("--traverse-npz-root", type=str,
                        default="/media/adam/vprdatasets/megaevent/brisbane_npz",
                        help="Root of the per-source frame trees: "
                             "<root>/<dataset>/<source>/<traverse>/frame_%%06d.npz.")
    # Comparison baselines (image datasets, or a traverse with --source)
    parser.add_argument("--method", type=str, default="megaevent",
                        choices=["megaevent", "sparse_event", "eventvlad", "eventgem",
                                 "spikevpr"],
                        help="Which VPR method to evaluate. The baselines are ports of "
                             "Event-LAB, Event-GeM and SpikeVPR, scored on identical data "
                             "and ground truth.")
    parser.add_argument("--eventlab-repo", type=str, default="/home/adam/repo/Event-LAB",
                        help="Event-LAB checkout supplying EventVLAD's networks and weights.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds for sparse_event's random pixel selection; results are "
                             "reported as mean +- std across them.")
    # Event-GeM args. Defaults are upstream's own (Event-GeM/main.py), except --eventgem-size,
    # which upstream has no equivalent of: it never resizes, because a traverse has one sensor
    # on both sides. Tokyo 24/7's splits do not (640x480 vs 480x854), so they have to meet on
    # a common grid the way sparse_event's do.
    parser.add_argument("--eventgem-repo", type=str, default="./external/eventgem",
                        help="Event-GeM checkout supplying SuperEvent and the re-ranker.")
    parser.add_argument("--eventgem-size", type=int, nargs=2, default=[240, 320],
                        metavar=("H", "W"),
                        help="Common MCTS grid for Event-GeM. The default is SuperEvent's own "
                             "geometry and a multiple of the 40 px its backbone requires.")
    parser.add_argument("--eventgem-top-k", type=int, default=50,
                        help="Candidates per query that Event-GeM re-ranks by homography.")
    parser.add_argument("--ransac-thresh", type=float, default=5.0,
                        help="Event-GeM RANSAC reprojection threshold, in pixels.")
    parser.add_argument("--inlier-weight", type=float, default=0.05,
                        help="Distance Event-GeM subtracts from a candidate per RANSAC inlier.")
    parser.add_argument("--match-filter", type=str, default="mutual",
                        choices=["mutual", "ratio"],
                        help="Keypoint correspondence filter before RANSAC.")
    parser.add_argument("--no-extra-pca", action="store_true",
                        help="Report only the shared PCA power, skipping a method's extra "
                             "whitening exponents (Event-GeM's full-whitening pca1). Each one "
                             "is a further re-rank pass and, on a large gallery, its own "
                             "keypoint store.")
    parser.add_argument("--match-ratio", type=float, default=0.8,
                        help="Lowe ratio, used only by --match-filter ratio.")
    # SpikeVPR args. Its forward pass runs in a separate pixi environment, because
    # spikingjelly is not in this one and the vendored clone's PyTorch is CPU-only; see
    # src/spikevpr_bridge.py.
    parser.add_argument("--spikevpr-model", type=str, default=None,
                        choices=["brisbane", "nsavp", "nyc"],
                        help="Which released SpikeVPR checkpoint to evaluate. Required for "
                             "--method spikevpr, and never defaulted: the three differ in "
                             "what they were trained on, which is the whole point of "
                             "running them cross-dataset.")
    parser.add_argument("--spikevpr-repo", type=str, default="./SpikeVPR",
                        help="SpikeVPR checkout supplying the model and its weights/.")
    parser.add_argument("--spikevpr-env", type=str, default="./envs/spikevpr",
                        help="pixi project the extractor runs in (CUDA torch + spikingjelly).")
    parser.add_argument("--spikevpr-max-events", type=int, default=None,
                        help="Render each frame from only the first N events of the window, "
                             "the way SpikeVPR's own Brisbane pipeline frames a slice with "
                             "ToFrame(event_count=15000). Off by default, so SpikeVPR sees "
                             "the same events every other method sees. It is a diagnostic "
                             "for Tokyo 24/7 above all, whose I2E saccades render a median "
                             "7.2-8.4 events/px against the 0.167 these checkpoints were "
                             "trained on — and the network normalises its input with "
                             "nothing but a frozen BatchNorm.")
    # Event stream parameters
    parser.add_argument("--no-hot-pixel", action="store_true",
                        help="If set, disables hot pixel removal.")
    parser.add_argument("--no-event-filter", action="store_true",
                        help="If set, disables background activity event filtering.")
    parser.add_argument("--event-filter-dt-ms", type=int, default=None,
                        help="Sets the time window in milliseconds for event filtering, default is the set dt-ms.")

    args = parser.parse_args()

    image_set = args.dataset in IMAGE_DATASETS
    if image_set:
        default_ref, default_query = IMAGE_SPLITS[args.dataset]
        args.ref = args.ref or default_ref
        args.query = args.query or default_query
    elif not (args.ref and args.query):
        parser.error(f"--ref and --query are required for {args.dataset}")
    if image_set and args.source:
        parser.error(f"--source applies to traverse datasets; {args.dataset} is an image "
                     f"set, where each file is already one independent event stream")
    if args.dataset == "nycevent" and args.method == "megaevent" and not args.ckpt:
        parser.error("nycevent MegaEvent inference requires --ckpt PATH")
    if args.method == "spikevpr" and not args.spikevpr_model:
        parser.error("--method spikevpr requires --spikevpr-model {brisbane,nsavp,nyc}")
    if args.spikevpr_model and args.method != "spikevpr":
        parser.error(f"--spikevpr-model applies to --method spikevpr, not {args.method}")
    if not image_set and not args.source and args.method != "megaevent":
        # Without --source the traverse path reads the HDF5 stream directly and renders one
        # representation for one model, so it has no method abstraction to dispatch on. With
        # --source the frames are materialised .npz and every method can consume them.
        parser.error(f"--method {args.method} needs either an image dataset "
                     f"({', '.join(sorted(IMAGE_DATASETS))}) or --source {{real,i2e}} to "
                     f"score {args.dataset} from pre-built frames")

    # Add the log file
    logger.remove()
    logpath = f"./logs/{args.dataset}/{args.ref}_{args.query}"
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    # via tqdm.write, not sys.stdout: it clears any live progress bar before writing and
    # redraws it after, so log lines never land on top of a bar. end="" because loguru
    # hands the sink an already-terminated message.
    logger.add(lambda m: tqdm.write(m, end=""), colorize=True,
               format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(f"{logpath}/{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")

    if image_set:
        # Each file is already one image, so there is no slicing window and no event stream
        # to filter. Say so rather than accepting the flags and quietly ignoring them.
        ignored = [name for name, given in (("--dt-ms", args.dt_ms != 50),
                                            ("--no-hot-pixel", args.no_hot_pixel),
                                            ("--no-event-filter", args.no_event_filter),
                                            ("--event-filter-dt-ms", args.event_filter_dt_ms))
                   if given]
        if ignored:
            logger.warning(f"{args.dataset} is an image set: {', '.join(ignored)} "
                           f"{'have' if len(ignored) > 1 else 'has'} no effect")
    else:
        # Update eventlab config and run data check
        eventlab(args)

    # Run the evaluation network
    inference(args)

if __name__ == "__main__":
    main()
