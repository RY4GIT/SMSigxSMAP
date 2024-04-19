import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import warnings
from MyLogger import getLogger

__author__ = "Ryoko Araki"
__contact__ = "raraki@ucsb.edu"
__copyright__ = "Copyright 2024, SMAP-drydown project, @RY4GIT"
__license__ = "MIT"
__status__ = "Dev"
__url__ = ""

# Create a logger
log = getLogger(__name__)


def get_filename(varname, EASE_row_index, EASE_column_index):
    """Get the filename of the datarod"""
    filename = f"{varname}_{EASE_row_index:03d}_{EASE_column_index:03d}.csv"
    return filename


def set_time_index(df, index_name="time"):
    """Set the datetime index to the pandas dataframe"""
    df[index_name] = pd.to_datetime(df[index_name])
    return df.set_index("time")


class Data:
    """Class that handles datarods (Precipitation, SM, PET data) for a EASE pixel"""

    def __init__(self, cfg, EASEindex) -> None:
        # _______________________________________________________________________________
        # Attributes

        # Read inputs
        self.cfg = cfg
        self.max_nodata_days = self.cfg.getint("EVENT_SEPARATION", "max_nodata_days")

        self.EASE_row_index = EASEindex[0]
        self.EASE_column_index = EASEindex[1]

        # Get the directory name
        self.data_dir = cfg["PATHS"]["data_dir"]
        self.datarods_dir = cfg["PATHS"]["datarods_dir"]

        # Get the start and end time of the analysis
        date_format = "%Y-%m-%d"
        self.start_date = datetime.strptime(cfg["EXTENT"]["start_date"], date_format)
        self.end_date = datetime.strptime(cfg["EXTENT"]["end_date"], date_format)

        # ______________________________________________________________________________
        # Get ancillary data
        self.theta_fc = self.get_anc_params()

        # _______________________________________________________________________________
        # Datasets
        _df = self.get_concat_datasets()
        self.df = self.calc_dSdt(_df)

    def get_concat_datasets(self):
        """Get datarods for each data variable, and concatinate them together to create a pandas dataframe"""

        # ___________________
        # Read each datasets
        sm = self.get_soil_moisture()
        pet = self.get_pet()
        p = self.get_precipitation()

        # ___________________
        # Concat all the datasets
        _df = pd.merge(sm, pet, how="outer", left_index=True, right_index=True)
        df = pd.merge(_df, p, how="outer", left_index=True, right_index=True)

        return df

    def get_dataframe(self, varname):
        """Get the pandas dataframe for a datarod of interest

        Args:
            varname (string): name of the variable: "SPL3SMP", "PET", "SPL4SMGP"

        Returns:
            dataframe: Return dataframe with datetime index, cropped for the timeperiod for a variable
        """

        fn = get_filename(
            varname,
            EASE_row_index=self.EASE_row_index,
            EASE_column_index=self.EASE_column_index,
        )
        _df = pd.read_csv(os.path.join(self.data_dir, self.datarods_dir, varname, fn))

        # Set time index and crop
        _df = set_time_index(_df, index_name="time")
        return _df[self.start_date : self.end_date]

    def get_soil_moisture(self, varname="SPL3SMP"):
        """Get a datarod of soil moisture data for a pixel"""

        # Get variable dataframe
        _df = self.get_dataframe(varname=varname)

        # Use retrieval flag to quality control the data
        condition_bad_data_am = (
            _df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 0.0
        ) & (_df["Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag"] != 8.0)
        condition_bad_data_pm = (
            _df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 0.0
        ) & (_df["Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm"] != 8.0)
        _df.loc[
            condition_bad_data_am, "Soil_Moisture_Retrieval_Data_AM_soil_moisture"
        ] = np.nan
        _df.loc[
            condition_bad_data_pm, "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm"
        ] = np.nan

        # If there is two different versions of 2015-03-31 data --- remove this
        df = _df.loc[~_df.index.duplicated(keep="first")]

        # Resample to regular time interval
        df = df.resample("D").asfreq()

        # Merge the AM and PM soil moisture data into one daily timeseries of data
        df["soil_moisture_daily"] = df[
            [
                "Soil_Moisture_Retrieval_Data_AM_soil_moisture",
                "Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm",
            ]
        ].mean(axis=1, skipna=True)

        # Get max and min values
        self.min_sm = df.soil_moisture_daily.min(skipna=True)
        self.max_sm = df.soil_moisture_daily.max(skipna=True)
        # Instead of actual max values, take the 95% percentile as max_sm # df.soil_moisture_daily.max(skipna=True)
        self.quantile = df.soil_moisture_daily.quantile(0.95)

        df["soil_moisture_daily_before_masking"] = df["soil_moisture_daily"].copy()
        # Mask out the timeseries when sm is larger than 90% percentile value
        df.loc[df["soil_moisture_daily"] > self.theta_fc, "soil_moisture_daily"] = (
            np.nan
        )

        return df

    def get_pet(self, varname="PET"):
        """Get a datarod of PET data for a pixel"""

        # Get variable dataframe
        _df = self.get_dataframe(varname=varname)

        # Drop unnccesary dimension
        _df = _df.drop(columns=["x", "y"])

        # Resample to regular time intervals
        return _df.resample("D").asfreq()

    def get_anc_params(self):
        """Get a datarod of PET data for a pixel"""

        # Get variable dataframe
        _df = pd.read_csv(
            os.path.join(self.data_dir, self.datarods_dir, "anc_info.csv")
        )

        # Drop unnccesary dimension
        matching_row = _df.loc[
            (_df["EASE_column_index"] == self.EASE_column_index)
            & (_df["EASE_row_index"] == self.EASE_row_index)
        ]

        fc = matching_row["theta_fc"].values[0]

        # Resample to regular time intervals
        return fc

    def get_precipitation(self, varname="SPL4SMGP"):
        """Get a datarod of precipitation data for a pixel"""

        # Get variable dataframe
        _df = self.get_dataframe(varname=varname)

        # Drop unnccesary dimension and change variable name
        _df = _df.drop(columns=["x", "y"]).rename(
            {"precipitation_total_surface_flux": "precip"}, axis="columns"
        )

        # Convert precipitation from kg/m2/s to mm/day -> 1 kg/m2/s = 86400 mm/day
        _df.precip = _df.precip * 86400

        # Resample to regular time interval
        return _df.resample("D").asfreq()

    def calc_dSdt(self, df):
        """Calculate d(Soil Moisture)/dt"""

        # Allow detecting soil moisture increment even if there is no SM data in between before/after rainfall event
        df["sm_for_dS_calc"] = (
            df["soil_moisture_daily_before_masking"].ffill().infer_objects(copy=False)
        )

        # Calculate dS
        df["dS"] = (
            df["sm_for_dS_calc"]
            .bfill(limit=self.max_nodata_days)
            .infer_objects(copy=False)
            .diff()
            .where(df["sm_for_dS_calc"].notnull())
            .replace(0, np.nan)
        )

        # Calculate dt
        nan_counts = (
            df["dS"]
            .isnull()
            .astype(int)
            .groupby(df["dS"].notnull().cumsum())
            .cumsum()
            .shift(1)
        )
        df["dt"] = nan_counts.fillna(0).infer_objects(copy=False).astype(int) + 1

        # Calculate dS/dt
        df["dSdt"] = df["dS"] / df["dt"]
        df.loc[df["soil_moisture_daily_before_masking"].isna(), "dSdt"] = np.nan
        df["dSdt"] = (
            df["dSdt"].ffill(limit=self.max_nodata_days).infer_objects(copy=False)
        )

        return df
