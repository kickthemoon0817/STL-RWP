import xarray as xr

ds_u_2017_2019 = xr.open_dataset(
    "./data/era5_2017_2019_u10n_v10n_shww.grib",
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"shortName": "u10n"},
        "indexpath": ""  # force a rebuild of the .idx if needed
    }
)
ds_v_2017_2019 = xr.open_dataset(
    "./data/era5_2017_2019_u10n_v10n_shww.grib",
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"shortName": "v10n"},
        "indexpath": ""
    }
)
ds_swh_2017_2019 = xr.open_dataset(
    "./data/era5_2017_2019_u10n_v10n_shww.grib",
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"shortName": "shww"},
        "indexpath": ""
    }
)

ds_u_2020_2021 = xr.open_dataset(
    "./data/era5_2020_2021_u10n_v10n_shww.grib",
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"shortName": "u10n"},
        "indexpath": ""
    }
)
ds_v_2020_2021 = xr.open_dataset(
    "./data/era5_2020_2021_u10n_v10n_shww.grib",
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"shortName": "v10n"},
        "indexpath": ""
    }
)
ds_swh_2020_2021 = xr.open_dataset(
    "./data/era5_2020_2021_u10n_v10n_shww.grib",
    engine="cfgrib",
    backend_kwargs={
        "filter_by_keys": {"shortName": "shww"},
        "indexpath": ""
    }
)

ds_u_all  = xr.concat([ds_u_2017_2019,  ds_u_2020_2021], dim="time")
ds_v_all = xr.concat([ds_v_2017_2019, ds_v_2020_2021], dim="time")
ds_swh_all = xr.concat([ds_swh_2017_2019, ds_swh_2020_2021], dim="time")

ds_u_all.to_netcdf("./data/era5_2017_2021_u10n_0.25deg.nc")
ds_v_all.to_netcodf("./data/era5_2017_2021_v10n_0.25deg.nc")
ds_swh_all.to_netcdf("./data/era5_2017_2021_shww_0.50deg.nc")
