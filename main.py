import argparse, os, time

from loguru import logger
from tqdm import tqdm
from src.inference import run
from src.eventlab import update_cfg

def eventlab(args):
    # Check if dataset exists, if not download it
    if not os.path.exists(f"{args.eventlab_dir}/{args.dataset}/{args.ref}") or not os.path.exists(f"{args.eventlab_dir}/{args.dataset}/{args.query}") or not os.path.exists(f"{args.eventlab_dir}/{args.dataset}/ground_truth/{args.ref}_{args.query}_GT.npy"):
        update_cfg(args)
    else:
        logger.info(f"Data exists at {args.eventlab_dir}/{args.dataset}")

def inference(args):
    logger.info(f"Running inference on {args.dataset}: DB {args.ref} <--> Q {args.query} @ {args.dt_ms} ms.")
    run(args)

def main():

    parser = argparse.ArgumentParser(description="Args for megaevent.")

    # Inference parameters
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["brisbane_event", "nsavp", "nycevent", "pitts", "tokyo"],
                        help="Sets the dataset to be used for inference.")
    parser.add_argument("--ref", "-r", type=str, required=True,
                        help="Sets the reference dataset for inference.")
    parser.add_argument("--query", "-q", type=str, required=True,
                        help="Sets the query dataset for inference.")
    parser.add_argument("--dt-ms", type=int, default=50,
                        help="Sets the time window in milliseconds for inference.")
    parser.add_argument("--model", "-m", type=str, default="s_salad_ft4", choices=["s_salad_ft4", "s_gem_ft4"],
                        help="Sets the model to be used for inference.")
    parser.add_argument("--feature-dir", type=str, default="./features",
                        help="Sets the directory to save features.")
    # Event-LAB args
    parser.add_argument("--eventlab-dir", type=str, default="./data",
                        help="Sets the Event-LAB default dataset directory.")
    # Event stream parameters
    parser.add_argument("--no-hot-pixel", action="store_true",
                        help="If set, disables hot pixel removal.")
    parser.add_argument("--no-event-filter", action="store_true",
                        help="If set, disables background activity event filtering.")
    parser.add_argument("--event-filter-dt-ms", type=int, default=None,
                        help="Sets the time window in milliseconds for event filtering, default is the set dt-ms.")

    args = parser.parse_args()

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

    # Update eventlab config and run data check
    eventlab(args)

    # Run the evaluation network
    inference(args)

if __name__ == "__main__":
    main()