# Assesment of Satellite-Derived Shorelines and beach slopes in Buenos Aires province (Argentina).  CoastSat and CoastSat.slope 

The objective of this set of notebooks is to obtain time series of satellite derived beach width and beach slopes, and compare it with in situ data obtained through beach profile measurements at six sites in Buenos Aires province in the period 2009-2018.The repository contains all input files, evaluation codes and output results.

This repo is an application and assessment of [CoastSat](https://github.com/kvos/CoastSat) and [CoastSat.slope](https://github.com/kvos/CoastSat.slope) tools, created by [K.Vos](https://github.com/kvos).

![figura 1 2](https://github.com/user-attachments/assets/34f638ac-c17d-442d-b3d1-fe3cf0407faf)

## Requirements

To run the notebooks it is necessary to install 

1.  [CoastSat](https://github.com/kvos/CoastSat) environment.
2.  FES2022 setup, the instructions to install it can be found [here](https://github.com/kvos/CoastSat/blob/master/doc/FES2022_setup.md).
3.  extra packages: [py_wave_runup](https://pypi.org/project/py-wave-runup/), [pymannkendall](https://pypi.org/project/pymannkendall/)

## Input Files

### SDS
1. Satellite-Derived Shorelines: There are two sets of shorelines extracted with the CoastSat tool:
      a. "shorelines_local_image_classificatio_model": Shorelines extracted with an image classifier trained with local imagery
      b. "shorelines_defoult_image_classificatio_model": Shorelines extracted with the default trained classifier, with sand color set in "default"
2. Shore-normal transects: store in SDS folder. These have the same starting point and orientation as the in situ beach profiles.

### Tide

3. tide_msas.csv: contains hourly sea level time series extracted from the “Modelling System for the Argentine Sea” (MSAS) model (Dinapoli et al. 2020).
4. tide nodes.geojson: geographic location from which the tidal time series of the FES2022 model are extracted.

### WindWaves

5. wind_waves.pkl:  3-h time series of Hs, Tp and Dir wind wave extracted from the global product Wavery.
6. wind_waves_dates.pkl: time associated with wind_waves.pkl

### In Situ data

7. in_situ_bw.pkl: Beach width time series refered to Mean Sea Level, extracted from Beach surveys were carried out using precise geometric leveling along transects perpendicular to the shore, extending to approximately 1 m deep
8. in_situ_slope_HAT-LAT.pkl: Face-beach slope computed from the profile segment that corresponds to the active beach, defined as the area between Highest Astronomical Tide (HAT) and Lowest Astronomical Tide (LAT)
9. in_situ_slope_MHHW-MLLW.pkl: beach slopes computed from the profile segment between the Mean Highest High Water (MHHW) and the Mean Lowest Low Water (MLLW)

## Notebooks

The following notebooks are available in this repo:

1. In situ data.ipynb: This code loads beach slopes and beach width data obtained from systematic surveys of beach profiles on multiple sites along the Buenos Aires Atlantic Coast: Punta Rasa (PR), Mar del Tuyú (MDT), Mar de Ajó (MDA), Nueva Atlantis (NA), Pinamar (PI), and Mar de las Pampas (MDP). Visualizes the results and obtains average slope for each site for later comparison with the one obtained from satellites.
2. CS_slope.ipynb: This code is an adaptation of the script by [K. Vos](https://github.com/kvos/CoastSat.slope).Calculates beach slopes along transects in 6 sites in Buenos Aires (Argentina) using shoreline data extracted by [CoastSat](https://github.com/kvos/CoastSat) and tide data from the FES2022 model.
3. CS_BWsat.ipynb: This notebook is an adaptation of the [CoastSat](https://github.com/kvos/CoastSat) tool and scripts [SDS_Benchmark](https://github.com/SatelliteShorelines/SDS_Benchmark). This notebook calculates the satellite-derived beach width (BWsat) for six sites in Buenos Aires Province, Argentina, using shoreline data extracted with CoastSat and shore-normal transects. The workflow involves tidal corrections to standardize the BWsat values to mean sea level using FES2022 tidal data and MSAS sea level data, followed by wave setup corrections with data from the Wavery product to account for wave effects. Results are saved for further analysis
4. Assessment.ipynb: This script evaluates the performance of satellite-derived shoreline data corrected with tide models and wave setup data. Specifically, it loads satellite-derived shoreline data corrected with the FES2022 and MSAS tidal models, with additional adjustments for wave setup using Wavery data (Obtained with the notebook 3. CS_BWsat). These corrected data sets are compared against beach width extracted from in situ surveys to assess their accuracy, in six sites of Buenos Aires providence. The script includes a comparison of beach slopes derived from satellite (Obtained with the notebook 2. CS_slope) data with slopes obtained from in situ measurements to evaluate the reliability of satellite-derived slope estimates (Obtained with the notebook 1. In situ data).
5. Slope sensitivity.ipynb: Sensitivity to beach slope variation in the tidal correction applied to BWsat time series is analyzed using MSAS data. Tidal correction is performed with different slope values varying from 0.01 to 0.2 in increments of 0.005.

   
