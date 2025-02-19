from Data import Data
from DrydownModel import DrydownModel
from EventSeparator import EventSeparator
from SMAPgrid import SMAPgrid
import warnings
from datetime import datetime
import os
import getpass
import pandas as pd
import logging
from MyLogger import getLogger

__author__ = "Ryoko Araki"
__contact__ = "raraki@ucsb.edu"
__copyright__ = "Copyright 2024, SMAP-drydown project, @RY4GIT"
__license__ = "MIT"
__status__ = "Dev"
__url__ = ""

# Create a logger
log = getLogger(__name__)


def create_output_dir(parent_dir):
    username = getpass.getuser()
    formatted_now = datetime.now().strftime("%Y-%m-%d")
    output_dir = rf"{parent_dir}/{username}_{formatted_now}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.info(f"Directory '{output_dir}' created.")
    return output_dir


class Agent:
    def __init__(self, cfg=None, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.smapgrid = SMAPgrid(cfg=self.cfg)
        self.target_EASE_idx = self.smapgrid.get_EASE_index_subset()
        self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]
        self.output_dir = create_output_dir(parent_dir=cfg["PATHS"]["output_dir"])

    def initialize(self):
        None

    def run(self, sample_EASE_index):
        """Run the analysis for one pixel

        Args:
            sample_EASE_index (list.shape[1,2]): a pair of EASE index, representing [0,0] the EASE row index (y, or latitude) and [0,1] EASE column index (x, or longitude)
        """

        try:
            # _______________________________________________________________________________________________
            # Get the sampling point attributes (EASE pixel)
            if self.verbose:
                log.info(
                    f"Currently processing pixel {sample_EASE_index}",
                )

            # _______________________________________________________________________________________________
            # Read dataset for a pixel
            data = Data(self.cfg, sample_EASE_index)

            # If there is no soil moisture data available for the pixel, skip the analysis
            if data.df.sm_masked.isna().all():
                warnings.warn(
                    f"No soil moisture data at the EASE pixel {sample_EASE_index}"
                )
                return None

            # _______________________________________________________________________________________________
            # Run the stormevent separation
            separator = EventSeparator(self.cfg, data)
            events = separator.separate_events(output_dir=self.output_dir)

            # If there is no drydown event detected for the pixel, skip the analysis
            # Check if there is SM data
            if not events:
                log.warning(f"No event drydown was detected at {sample_EASE_index}")
                return None

            log.info(
                f"Event separation success at {sample_EASE_index}: {len(events)} events detected"
            )

            # _______________________________________________________________________________________________
            # Execute the main analysis --- fit drydown models
            drydown_model = DrydownModel(self.cfg, data, events)
            drydown_model.fit_models(output_dir=self.output_dir)

            results_df = drydown_model.return_result_df()

            log.info(
                f"Drydown model analysis completed at {sample_EASE_index}: {len(results_df)}/{len(events)} events fitted"
            )

            return results_df

        except Exception as e:
            print(f"Error in thread: {sample_EASE_index}")
            print(f"Error message: {str(e)}")

    def finalize(self, results):
        """Finalize the analysis from all the pixels

        Args:
            results (list): concatinated results returned from serial/multi-threadding analysis
        """
        self.save_to_csv(results)
        self.save_config()
        # self.smapgrid.remap_results(df_results)
        # self.smapgrid.plot_remapped_results(da)

    def save_to_csv(self, results):
        if len(results) > 1:
            try:
                df = pd.concat(results)
            except:
                df = results
        else:
            df = results
        df.to_csv(os.path.join(self.output_dir, "all_results.csv"))
        return df

    def save_config(self):
        with open(os.path.join(self.output_dir, "config.ini"), "w") as cfg_file:
            self.cfg.write(cfg_file)
