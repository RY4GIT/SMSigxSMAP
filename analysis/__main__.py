import multiprocessing as mp
from configparser import ConfigParser
import time

from Agent import Agent
from MyLogger import getLogger

__author__ = "Ryoko Araki"
__contact__ = "raraki@ucsb.edu"
__copyright__ = "Copyright 2024, SMAP-drydown project, @RY4GIT"
__license__ = "MIT"
__status__ = "Dev"
__url__ = ""

# Create a logger
log = getLogger(__name__)


def main():
    """Main execution script ot run the drydown analysis"""
    start = time.perf_counter()
    log.info("--- Initializing the model ---")

    # _______________________________________________________________________________________________
    # Read config
    cfg = ConfigParser()
    cfg.read("config.ini")

    # Initiate agent
    agent = Agent(cfg=cfg)
    agent.initialize()

    # _______________________________________________________________________________________________
    # Define serial/parallel mode
    run_mode = cfg.get("MODEL", "run_mode")
    log.info(f"--- Analysis started with {run_mode} mode ---")

    # _______________________________________________________________________________________________
    # Verbose models to run
    log.info(f"Running the following models:")
    if cfg.getboolean("MODEL", "tau_exp_model"):
        log.info(f"Tau-based Exponential model")
    if cfg.getboolean("MODEL", "exp_model"):
        log.info(f"Exponential model")
    if cfg.getboolean("MODEL", "q_model"):
        log.info(f"q model")
    if cfg.getboolean("MODEL", "sigmoid_model"):
        log.info(f"Sigmoid model")

    # Run the model
    if run_mode == "serial":
        results = agent.run(
            [181, 513]
        )  # Pick your EASE_row_index and EASE_column_index of interest
    elif run_mode == "parallel":
        nprocess = cfg.getint("MULTIPROCESSING", "nprocess")
        with mp.Pool(nprocess) as pool:
            results = list(pool.imap(agent.run, agent.target_EASE_idx))
        pool.close()
        pool.join()
    else:
        log.info(
            "run_mode in config is invalid: should be either 'serial' or 'parallel'"
        )

    # _______________________________________________________________________________________________
    # Finalize the model
    log.info(f"--- Finished analysis ---")

    if run_mode == "serial":
        if results.empty:
            log.info("No results are returned")
        else:
            try:
                agent.finalize(results)
            except NameError:
                log.info("No results are returned")

    elif run_mode == "parallel":
        if not results:
            log.info("No results are returned")
        else:
            try:
                agent.finalize(results)
            except NameError:
                log.info("No results are returned")

    end = time.perf_counter()
    log.info(f"Run took : {(end - start):.6f} seconds")


if __name__ == "__main__":
    main()
