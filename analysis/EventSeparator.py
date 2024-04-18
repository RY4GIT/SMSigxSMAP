import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Event import Event
import warnings
import threading
from MyLogger import getLogger, modifyLogger
import logging
from utils import is_true

__author__ = "Ryoko Araki"
__contact__ = "raraki@ucsb.edu"
__copyright__ = "Copyright 2024, SMAP-drydown project, @RY4GIT"
__license__ = "MIT"
__status__ = "Dev"
__url__ = ""

# Create a logger
log = getLogger(__name__)


class ThreadNameHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            # Add thread name to the log message
            record.threadName = threading.current_thread().name
            super(ThreadNameHandler, self).emit(record)
        except Exception:
            self.handleError(record)


class EventSeparator:
    def __init__(self, cfg, Data):
        self.cfg = cfg
        self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]
        self.use_rainfall = cfg["MODEL"]["use_rainfall"].lower() in [
            "true",
            "yes",
            "1",
        ]  # is_true(cfg["MODEL"]["use_rainfall"])
        self.plot = cfg["MODEL"]["plot_results"].lower() in [
            "true",
            "yes",
            "1",
        ]  # is_true(cfg["MODEL"]["plot_results"])

        self.data = Data
        self.init_params()

        current_thread = threading.current_thread()
        current_thread.name = (
            f"[{self.data.EASE_row_index},{self.data.EASE_column_index}]"
        )
        self.thread_name = current_thread.name

        # # Not working at the moment ...
        # custom_handler = ThreadNameHandler()
        # log = modifyLogger(name=__name__, custom_handler=custom_handler)

    def init_params(self):
        self.precip_thresh = self.cfg.getfloat("EVENT_SEPARATION", "precip_thresh")
        self.target_rmsd = self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd")
        _noise_thresh = (self.data.max_sm - self.data.min_sm) * self.cfg.getfloat(
            "EVENT_SEPARATION", "increment_thresh_fraction"
        )
        self.noise_thresh = np.minimum(_noise_thresh, self.target_rmsd * 2)
        self.dS_thresh = self.target_rmsd * 2
        self.minimium_consective_days = self.cfg.getint(
            "EVENT_SEPARATION", "minimium_consective_days"
        )
        self.max_nodata_days = self.cfg.getint("EVENT_SEPARATION", "max_nodata_days")

    def separate_events(self, output_dir):
        """Separate soil moisture timeseries into events"""
        self.output_dir = output_dir

        self.identify_event_starts()
        self.adjust_event_starts_1()
        self.adjust_event_starts_2()
        self.identify_event_ends()

        self.events_df = self.create_event_dataframe()
        if self.events_df.empty:
            return None

        self.filter_events(self.minimium_consective_days)
        self.events = self.create_event_instances(self.events_df)

        if self.plot:
            self.plot_events()

        return self.events

    def identify_event_starts(self):
        """Identify the start date of the event"""
        # The event starts where negative increament of soil mositure follows the positive increment of soil moisture
        negative_increments = (self.data.df.dSdt < 0) | (np.isnan(self.data.df.dS))
        positive_increments = self.data.df.dSdt > self.dS_thresh
        self.data.df["event_start"] = negative_increments.shift(-1).fillna(
            False
        ).infer_objects(copy=False).astype(bool) & positive_increments.fillna(
            False
        ).infer_objects(
            copy=False
        ).astype(
            bool
        )

    def adjust_event_starts_1(self):
        event_start_idx = self.data.df["event_start"][self.data.df["event_start"]].index
        for _, event_start_date in enumerate(event_start_idx):

            should_break = False
            for j in range(
                0, self.max_nodata_days + 1
            ):  # Look back up to 3 timesteps to seek for sm value which is not nan, or start of the precip event
                current_date = event_start_date - pd.Timedelta(days=j)

                if self.use_rainfall:
                    if self.data.df.loc[current_date].precip > self.precip_thresh:
                        if not np.isnan(
                            self.data.df.loc[
                                current_date
                            ].soil_moisture_daily_before_masking
                        ):
                            self.data.df.loc[event_start_date, "event_start"] = False
                            self.data.df.loc[current_date, "event_start"] = True
                        should_break = True
                if should_break:
                    break

                # If dS > 0 and SM value is not nan, use that
                if (self.data.df.loc[current_date].dSdt > self.dS_thresh) & ~np.isnan(
                    self.data.df.loc[current_date].soil_moisture_daily_before_masking
                ):
                    self.data.df.loc[event_start_date, "event_start"] = False
                    self.data.df.loc[current_date, "event_start"] = True
                    break

                if (np.isnan(self.data.df.loc[current_date].dSdt)) or (
                    self.data.df.loc[current_date].dSdt < -1 * self.noise_thresh
                ):
                    break

    def adjust_event_starts_2(self):

        # If the soil moisture data at the beginning of the event is no data or exceeds threshold, look for subsequent available data
        condition_mask = (
            pd.isna(self.data.df["soil_moisture_daily"]) & self.data.df["event_start"]
        )
        event_start_nan_idx = self.data.df.index[condition_mask]

        for i, event_start_date in enumerate(event_start_nan_idx):
            current_date = event_start_date
            if (
                self.data.df.loc[event_start_date].soil_moisture_daily_before_masking
                > self.data.max_sm
            ) | (
                np.isnan(
                    self.data.df.loc[
                        event_start_date
                    ].soil_moisture_daily_before_masking
                )
            ):
                current_date += pd.Timedelta(days=1)
                self.data.df.loc[event_start_date, "event_start"] = False
                self.data.df.loc[current_date, "event_start"] = True

                if (
                    self.data.df.loc[current_date].soil_moisture_daily_before_masking
                    > self.data.max_sm
                ) | (
                    np.isnan(
                        self.data.df.loc[
                            current_date
                        ].soil_moisture_daily_before_masking
                    )
                ):
                    current_date += pd.Timedelta(days=1)
                    self.data.df.loc[
                        current_date - pd.Timedelta(days=1), "event_start"
                    ] = False
                    self.data.df.loc[current_date, "event_start"] = True

                    if (
                        self.data.df.loc[
                            current_date
                        ].soil_moisture_daily_before_masking
                        > self.data.max_sm
                    ) | (
                        np.isnan(
                            self.data.df.loc[
                                current_date
                            ].soil_moisture_daily_before_masking
                        )
                    ):
                        self.data.df.loc[
                            current_date - pd.Timedelta(days=1), "event_start"
                        ] = False
                        self.data.df.loc[current_date, "event_start"] = True

        if self.use_rainfall:
            event_start_idx = self.data.df["event_start"][
                self.data.df["event_start"]
            ].index
            for _, event_start_date in enumerate(event_start_idx):
                current_date = event_start_date + pd.Timedelta(days=1)
                if self.data.df.loc[current_date].precip > self.precip_thresh:
                    current_date += pd.Timedelta(days=1)
                    self.data.df.loc[event_start_date, "event_start"] = False
                    self.data.df.loc[
                        current_date - pd.Timedelta(days=1), "event_start"
                    ] = False
                    self.data.df.loc[current_date, "event_start"] = True

                    if self.data.df.loc[current_date].precip > self.precip_thresh:
                        current_date += pd.Timedelta(days=1)
                        self.data.df.loc[event_start_date, "event_start"] = False
                        self.data.df.loc[
                            current_date - pd.Timedelta(days=1), "event_start"
                        ] = False
                        self.data.df.loc[current_date, "event_start"] = True

    def identify_event_ends(self):
        self.data.df["event_end"] = np.zeros(len(self.data.df), dtype=bool)
        self.event_start_idx = pd.Series(
            self.data.df["event_start"][self.data.df["event_start"]].index
        )
        self.event_end_idx = pd.Series([pd.NaT] * len(self.data.df["event_start"]))
        record_last_date = self.data.df.index.values[-1]

        for i, event_start_date in enumerate(self.event_start_idx):
            remaining_records = record_last_date - event_start_date
            count_nan_days = 0
            should_break = False

            for j in range(1, remaining_records.days):
                current_date = event_start_date + pd.Timedelta(days=j)

                if np.isnan(self.data.df.loc[current_date].soil_moisture_daily):
                    count_nan_days += 1
                else:
                    count_nan_days = 0

                if self.data.df.loc[current_date].precip >= self.precip_thresh:
                    update_date = current_date - pd.Timedelta(days=1)
                    update_arg = "precipitation exceeds threshold"
                    should_break = True

                if count_nan_days > self.max_nodata_days:
                    update_date = current_date
                    update_arg = "too many consective nans"
                    should_break = True

                if (self.data.df.loc[current_date].dS >= self.noise_thresh) & (
                    ~np.isnan(self.data.df.loc[current_date].soil_moisture_daily)
                ):
                    # Any positive increment smaller than 5% of the observed range of soil moisture at the site is excluded (if there is not precipitation) if it would otherwise truncate a drydown.
                    update_date = current_date - pd.Timedelta(days=1)
                    update_arg = "dS increment exceed the noise threshold"
                    should_break = True

                if should_break:
                    self.data.df.loc[update_date, "event_end"] = True
                    self.event_end_idx[i] = update_date
                    break

        # create a new column for event_end
        self.data.df["dSdt(t-1)"] = self.data.df.dSdt.shift(+1)

    def create_event_dataframe(self):

        # Create a new DataFrame with each row containing a list of soil moisture values between each pair of event_start and event_end
        event_data = [
            {
                "event_start": start_index,
                "event_end": end_index,
                "min_sm": self.data.min_sm,
                "max_sm": self.data.max_sm,
                "theta_fc": self.data.theta_fc,
                "soil_moisture_daily": list(
                    self.data.df.loc[
                        start_index:end_index, "soil_moisture_daily"
                    ].values
                ),
                "soil_moisture_daily_before_masking": list(
                    self.data.df.loc[
                        start_index:end_index, "soil_moisture_daily_before_masking"
                    ].values
                ),
                "precip": list(
                    self.data.df.loc[start_index:end_index, "precip"].values
                ),
                "PET": list(self.data.df.loc[start_index:end_index, "pet"].values),
                "delta_theta": self.data.df.loc[start_index, "dSdt(t-1)"],
            }
            for start_index, end_index in zip(self.event_start_idx, self.event_end_idx)
        ]
        return pd.DataFrame(event_data)

    def filter_events(self, min_consecutive_days=5):
        self.events_df = self.events_df[
            self.events_df["soil_moisture_daily_before_masking"].apply(
                lambda x: pd.notna(x).sum()
            )
            >= min_consecutive_days
        ].copy()
        self.events_df.reset_index(drop=True, inplace=True)

    def create_event_instances(self, events_df):
        """Create a list of Event instances for easier handling of data for DrydownModel class"""
        event_instances = [
            Event(index, row.to_dict()) for index, row in events_df.iterrows()
        ]
        return event_instances

    def plot_events(self):
        fig, (ax11, ax12) = plt.subplots(2, 1, figsize=(20, 5))

        self.data.df.soil_moisture_daily_before_masking.plot(ax=ax11, alpha=0.5)
        ax11.scatter(
            self.data.df.soil_moisture_daily_before_masking[
                self.data.df["event_start"]
            ].index,
            self.data.df.soil_moisture_daily_before_masking[
                self.data.df["event_start"]
            ].values,
            color="orange",
            alpha=0.5,
        )
        ax11.scatter(
            self.data.df.soil_moisture_daily_before_masking[
                self.data.df["event_end"]
            ].index,
            self.data.df.soil_moisture_daily_before_masking[
                self.data.df["event_end"]
            ].values,
            color="orange",
            marker="x",
            alpha=0.5,
        )
        self.data.df.precip.plot(ax=ax12, alpha=0.5)

        # Save results
        filename = f"{self.data.EASE_row_index:03d}_{self.data.EASE_column_index:03d}_eventseparation.png"
        output_dir2 = os.path.join(self.output_dir, "plots")
        if not os.path.exists(output_dir2):
            # Use a lock to ensure only one thread creates the directory
            with threading.Lock():
                # Check again if the directory was created while waiting
                if not os.path.exists(output_dir2):
                    os.makedirs(output_dir2)

        fig.savefig(os.path.join(output_dir2, filename))
        plt.close()

    # def adjust_event_starts_a(self):
    #     """In case the soil moisture data is nan on the initial event_start dates, look for data in the previous timesteps"""

    #     # Get the event start dates
    #     event_start_idx = self.data.df["event_start"][self.data.df["event_start"]].index

    #     # Loop for event start dates
    #     for i, event_start_date in enumerate(event_start_idx):
    #         for j in range(
    #             0, self.minimum_nodata_days
    #         ):  # Look back up to 6 timesteps to seek for sm value which is not nan
    #             current_date = event_start_date - pd.Timedelta(days=j)

    #             # If rainfall exceeds threshold, stop there
    #             if self.data.df.loc[current_date].precip > self.precip_thresh:
    #                 # If SM value IS NOT nap.nan, update the event start date value to this timestep
    #                 if not np.isnan(
    #                     self.data.df.loc[
    #                         current_date
    #                     ].soil_moisture_daily_before_masking
    #                 ):
    #                     update_date = current_date
    #                 # If SM value IS nap.nan, don't update the event start date value
    #                 else:
    #                     update_date = event_start_date
    #                 break

    #             # If dS > 0, stop there
    #             if self.data.df.loc[current_date].dS > 0:
    #                 update_date = event_start_date
    #                 break

    #             # If reached to the NON-nan SM value, update start date value to this timestep
    #             if ((i - j) >= 0) or (
    #                 not np.isnan(
    #                     self.data.df.loc[
    #                         current_date
    #                     ].soil_moisture_daily_before_masking
    #                 )
    #             ):
    #                 update_date = current_date
    #                 break

    #         # Update the startdate timestep
    #         self.data.df.loc[event_start_date, "event_start"] = False
    #         self.data.df.loc[update_date, "event_start"] = True

    # def adjust_event_starts_b(self):
    #     """In case the soil moisture data is nan on the initial event_start dates, look for data in the later timesteps"""

    #     # Get the event start dates
    #     event_start_idx_2 = pd.isna(
    #         self.data.df["soil_moisture_daily_before_masking"][
    #             self.data.df["event_start"]
    #         ]
    #     ).index

    #     for i, event_start_date in enumerate(event_start_idx_2):
    #         update_date = event_start_date
    #         for j in range(
    #             0, self.max_nodata_days
    #         ):  # Look ahead up to 6 timesteps to seek for sm value which is not nan, or start of the precip event
    #             current_date = event_start_date + pd.Timedelta(days=j)
    #             # If Non-nan SM value is detected, update start date value to this timstep
    #             if current_date > self.data.end_date:
    #                 update_date = current_date
    #                 break

    #             if not pd.isna(
    #                 self.data.df.loc[current_date].soil_moisture_daily_before_masking
    #             ):
    #                 update_date = current_date
    #                 break

    #         self.data.df.loc[event_start_date, "event_start"] = False
    #         self.data.df.loc[update_date, "event_start"] = True

    # def identify_event_ends_a(self):
    #     """Detect the end of a storm event"""

    #     # Initialize
    #     num_events = self.data.df.shape[0]
    #     self.data.df["event_end"] = np.zeros(len(self.data.df), dtype=bool)
    #     event_start_idx = self.data.df["event_start"][self.data.df["event_start"]].index
    #     for i, event_start_date in enumerate(event_start_idx):
    #         for j in range(1, len(self.data.df)):
    #             current_date = event_start_date + pd.Timedelta(days=j)

    #             if current_date > self.data.end_date:
    #                 break

    #             if np.isnan(
    #                 self.data.df.loc[current_date].soil_moisture_daily_before_masking
    #             ):
    #                 continue

    #             if (
    #                 (self.data.df.loc[current_date].dSdt >= self.noise_thresh)
    #                 or (self.data.df.loc[current_date].precip > self.precip_thresh)
    #             ) or self.data.df.loc[current_date].event_start:
    #                 # Any positive increment smaller than 5% of the observed range of soil moisture at the site is excluded (if there is not precipitation) if it would otherwise truncate a drydown.
    #                 self.data.df.loc[current_date, "event_end"] = True
    #                 break
    #             else:
    #                 None

    #     # create a new column for event_end
    #     self.data.df["dSdt(t-1)"] = self.data.df.dSdt.shift(+1)
