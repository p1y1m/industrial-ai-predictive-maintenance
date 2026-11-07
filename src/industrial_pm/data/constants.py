from __future__ import annotations

NASA_CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
NASA_DATASET_CITATION = (
    "A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, "
    "NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA."
)

ID_COLUMNS = ["unit_id", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
CMAPSS_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
SUPPORTED_SUBSETS = ("FD001", "FD002", "FD003", "FD004")

# Standard C-MAPSS sensor names and units from the dataset documentation/paper.
SENSOR_METADATA = {
    "sensor_1": ("T2", "Total temperature at fan inlet", "°R"),
    "sensor_2": ("T24", "Total temperature at LPC outlet", "°R"),
    "sensor_3": ("T30", "Total temperature at HPC outlet", "°R"),
    "sensor_4": ("T50", "Total temperature at LPT outlet", "°R"),
    "sensor_5": ("P2", "Pressure at fan inlet", "psia"),
    "sensor_6": ("P15", "Total pressure in bypass-duct", "psia"),
    "sensor_7": ("P30", "Total pressure at HPC outlet", "psia"),
    "sensor_8": ("Nf", "Physical fan speed", "rpm"),
    "sensor_9": ("Nc", "Physical core speed", "rpm"),
    "sensor_10": ("epr", "Engine pressure ratio", "—"),
    "sensor_11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "sensor_12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "sensor_13": ("NRf", "Corrected fan speed", "rpm"),
    "sensor_14": ("NRc", "Corrected core speed", "rpm"),
    "sensor_15": ("BPR", "Bypass ratio", "—"),
    "sensor_16": ("farB", "Burner fuel-air ratio", "—"),
    "sensor_17": ("htBleed", "Bleed enthalpy", "—"),
    "sensor_18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed", "%"),
    "sensor_20": ("W31", "HPT coolant bleed", "lbm/s"),
    "sensor_21": ("W32", "LPT coolant bleed", "lbm/s"),
}
