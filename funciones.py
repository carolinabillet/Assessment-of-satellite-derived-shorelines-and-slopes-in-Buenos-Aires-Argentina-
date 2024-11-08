

# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime, timedelta
import pytz
import pdb
from matplotlib import gridspec
from scipy import interpolate
from scipy import stats
from datetime import timedelta

# other modules
import skimage.transform as transform
from scipy import interpolate
from scipy import stats



def reorganizar_datos_in_situ(datos_lupe, sitio):
    # Obtener las fechas y distancias del sitio
    date = pd.to_datetime(datos_lupe[sitio]['date'])
    bw_ins = datos_lupe[sitio]['bw_ins']
    
    # Crear un DataFrame con las fechas y distancias
    df = pd.DataFrame({'date': date, 'BW': bw_ins})
    
    # Asegurarse de que las fechas estén como índice
    df = df.set_index('date')
    
    return df






def compare_timeseries(dates_sat1, dates_lupe, nm, lupe, keys, settings, satname_all):
    """
    Compares time-series by interpolating the satellite data around the ground truth data.
    Processes multiple transects specified by a list of keys. Lo hace con cada sat por separado
	y ademas saca los estadísticos en un dict
    """
    # Initialize dictionaries to store results and statistics
    results = {key: {} for key in keys}
    stats_dict = {key: {} for key in keys}

    for key in keys:
        if key not in lupe.keys() or key not in nm.keys():
            print(f'Transect name {key} does not exist in the provided dictionaries')
            continue

        dates_sat = dates_sat1[key]
        dates_sat = pd.to_datetime(dates_sat)
        dates_sat_naive = dates_sat.tz_localize(None)
        dates_sat_series = pd.Series(dates_sat_naive)

        # Remove NaNs from satellite data
        chain_sat_dm = np.array(nm[key])
        idx_nan_sat = np.isnan(chain_sat_dm)
        dates_nonans_sat = [dates_sat_series[k] for k in np.where(~idx_nan_sat)[0]]
        chain_nonans_sat = chain_sat_dm[~idx_nan_sat]
        satnames_nonans = [satname_all[key][k] for k in np.where(~idx_nan_sat)[0]]

        # Remove NaNs from in situ data
        chain_sur_dm = np.array(lupe[key])
        idx_nan_sur = np.isnan(chain_sur_dm)
        dates_sur = [dates_lupe[k] for k in np.where(~idx_nan_sur)[0]]
        chain_sur_dm = chain_sur_dm[~idx_nan_sur]


        # Interpolate surveyed data around satellite data based on the parameters (min_days and max_days)
        chain_int, dates_int, sat_int, idx_int = [], [], [], []
        for k, date in enumerate(dates_sur):
            # Compute the days distance for each satellite date
            days_diff = np.array([(_ - date).days for _ in dates_nonans_sat])
            # If nothing within max_days put a nan
            if np.min(np.abs(days_diff)) <= settings['max_days']:
                # If a point within min_days, take that point (no interpolation)
                if np.min(np.abs(days_diff)) < settings['min_days']:
                    idx_closest = np.where(np.abs(days_diff) == np.min(np.abs(days_diff)))[0][0]
                    chain_int.append(chain_nonans_sat[idx_closest])
                    dates_int.append(date)
                    sat_int.append(satnames_nonans[idx_closest])
                    idx_int.append(k)
                else:  # Otherwise, between min_days and max_days, interpolate between the 2 closest points
                    if sum(days_diff < 0) == 0 or sum(days_diff > 0) == 0:
                        continue
                    idx_after = np.where(days_diff > 0)[0][0]
                    idx_before = idx_after - 1
                    x = [dates_nonans_sat[idx_before].toordinal(), dates_nonans_sat[idx_after].toordinal()]
                    y = [chain_nonans_sat[idx_before], chain_nonans_sat[idx_after]]
                    f = interpolate.interp1d(x, y, bounds_error=True)
                    try:
                        chain_int.append(float(f(date.toordinal())))
                        dates_int.append(date)
                        sat_int.append(satnames_nonans[idx_before])
                        idx_int.append(k)
                    except:
                        continue
       
        if len(chain_int) == 0:
            print(f'Not enough data points for comparison at transect {key}')
            plt.close(fig)
            results[key] = {'chain_sat': [], 'chain_sur': [], 'satnames': [], 'fig': fig}
            stats_dict[key] = {'rmse': None, 'bias': None, 'mean': None, 'std': None, 'r2': None}
            continue

        # Remove NaNs again
        chain_int = np.array(chain_int)
        idx_nan = np.isnan(chain_int)
        chain_sat = chain_int[~idx_nan]
        chain_sur = chain_sur_dm[np.array(idx_int)][~idx_nan]
        dates_sat = [dates_int[k] for k in np.where(~idx_nan)[0]]
        satnames = [sat_int[k] for k in np.where(~idx_nan)[0]]

        # Make sure there's enough data points to compute the metrics
        if len(chain_sat) < 8 or len(chain_sur) < 8:
            print(f'Not enough data points for comparison at transect {key}')
            plt.close(fig)
            results[key] = {'chain_sat': [], 'chain_sur': [], 'satnames': [], 'fig': fig}
            stats_dict[key] = {'rmse': None, 'bias': None, 'mean': None, 'std': None, 'r2': None}
            continue

        # Error statistics
        slope, intercept, rvalue, pvalue, std_err = stats.linregress(chain_sur, chain_sat)
        R2 = rvalue**2
        chain_error = chain_sat - chain_sur
        rmse = np.sqrt(np.mean((chain_error)**2))
        mean = np.mean(chain_error)
        std = np.std(chain_error)
        q90 = np.percentile(np.abs(chain_error), 90)
        gof = 1 - np.sum((chain_sat - chain_sur)**2) / (np.sum((np.abs(chain_sat - np.mean(chain_sur)) + np.abs(chain_sur - np.mean(chain_sur)))**2))

        # Store statistics
        stats_dict[key] = {
            'rmse': rmse,
            'bias': mean,
            #'mean': mean,
            'std': std,
            'r2': R2
        }

 
        # Store results
        results[key] = {
            'chain_sat': chain_sat,
            'chain_sur': chain_sur,
            'satnames': satnames,

        }

    return results, stats_dict
def compare_timeseries2(dates_sat1, dates_lupe, nm, lupe, keys, settings, satname_all):
    """
    Compares time-series by interpolating the satellite data around the ground truth data.
    Processes multiple transects specified by a list of keys. Lo hace con cada sat por separado
	y ademas saca los estadísticos en un dict
    """
    # Initialize dictionaries to store results and statistics
    results = {key: {} for key in keys}
    stats_dict = {key: {} for key in keys}

    for key in keys:
        if key not in lupe.keys() or key not in nm.keys():
            print(f'Transect name {key} does not exist in the provided dictionaries')
            continue

        dates_sat = dates_sat1[key]
        dates_sat = pd.to_datetime(dates_sat)
        dates_sat_naive = dates_sat.tz_localize(None)
        dates_sat_series = pd.Series(dates_sat_naive)

        # Remove NaNs from satellite data
        chain_sat_dm = np.array(nm[key])
        idx_nan_sat = np.isnan(chain_sat_dm)
        dates_nonans_sat = [dates_sat_series[k] for k in np.where(~idx_nan_sat)[0]]
        chain_nonans_sat = chain_sat_dm[~idx_nan_sat]
        satnames_nonans = [satname_all[key][k] for k in np.where(~idx_nan_sat)[0]]

        # Remove NaNs from in situ data
        chain_sur_dm = np.array(lupe[key])
        idx_nan_sur = np.isnan(chain_sur_dm)
        dates_sur = [dates_lupe[k] for k in np.where(~idx_nan_sur)[0]]
        chain_sur_dm = chain_sur_dm[~idx_nan_sur]

        # Plot the time-series
        fig = plt.figure(figsize=[15, 8], tight_layout=True)
        gs = gridspec.GridSpec(2, 3)
        ax0 = fig.add_subplot(gs[0, :])
        ax0.grid(which='major', linestyle=':', color='0.5')
        ax0.plot(dates_sur, chain_sur_dm, 'C1-', label='In situ')
        ax0.plot(dates_nonans_sat, chain_nonans_sat, 'C0-', label='Satellite')
        date_start = np.max([dates_sur[0], dates_nonans_sat[0]])
        date_last = np.min([dates_sur[-1], dates_nonans_sat[-1]])
        ax0.set(title=f'Transect {key}', xlim=[date_start - timedelta(days=30), date_last + timedelta(days=30)])
        ax0.legend(ncol=2)

        # Interpolate surveyed data around satellite data based on the parameters (min_days and max_days)
        chain_int, dates_int, sat_int, idx_int = [], [], [], []
        for k, date in enumerate(dates_sur):
            # Compute the days distance for each satellite date
            days_diff = np.array([(_ - date).days for _ in dates_nonans_sat])
            # If nothing within max_days put a nan
            if np.min(np.abs(days_diff)) <= settings['max_days']:
                # If a point within min_days, take that point (no interpolation)
                if np.min(np.abs(days_diff)) < settings['min_days']:
                    idx_closest = np.where(np.abs(days_diff) == np.min(np.abs(days_diff)))[0][0]
                    chain_int.append(chain_nonans_sat[idx_closest])
                    dates_int.append(date)
                    sat_int.append(satnames_nonans[idx_closest])
                    idx_int.append(k)
                else:  # Otherwise, between min_days and max_days, interpolate between the 2 closest points
                    if sum(days_diff < 0) == 0 or sum(days_diff > 0) == 0:
                        continue
                    idx_after = np.where(days_diff > 0)[0][0]
                    idx_before = idx_after - 1
                    x = [dates_nonans_sat[idx_before].toordinal(), dates_nonans_sat[idx_after].toordinal()]
                    y = [chain_nonans_sat[idx_before], chain_nonans_sat[idx_after]]
                    f = interpolate.interp1d(x, y, bounds_error=True)
                    try:
                        chain_int.append(float(f(date.toordinal())))
                        dates_int.append(date)
                        sat_int.append(satnames_nonans[idx_before])
                        idx_int.append(k)
                    except:
                        continue
        ax0.plot(dates_sur, chain_sur_dm, 'C1o', mfc='none', ms=4)
        ax0.plot(dates_nonans_sat, chain_nonans_sat, 'C0o', mfc='none', ms=4)
        ax0.plot(dates_int, chain_int, 'ko', mfc='none', ms=4)

        ax0.set_title(f'Transect {key} - n_sur = {len(dates_sur)} - n_sat = {len(dates_nonans_sat)} - n_int = {len(dates_int)}')

        if len(chain_int) == 0:
            print(f'Not enough data points for comparison at transect {key}')
            plt.close(fig)
            results[key] = {'chain_sat': [], 'chain_sur': [], 'satnames': [],'dates_int': [],'dates_sur': [], 'fig': fig}
            stats_dict[key] = {'rmse': None, 'bias': None, 'mean': None, 'std': None, 'r2': None}
            continue

        # Remove NaNs again
        chain_int = np.array(chain_int)
        idx_nan = np.isnan(chain_int)
        chain_sat = chain_int[~idx_nan]
        chain_sur = chain_sur_dm[np.array(idx_int)][~idx_nan]
        dates_sat = [dates_int[k] for k in np.where(~idx_nan)[0]]
        satnames = [sat_int[k] for k in np.where(~idx_nan)[0]]

        # Make sure there's enough data points to compute the metrics
        if len(chain_sat) < 8 or len(chain_sur) < 8:
            print(f'Not enough data points for comparison at transect {key}')
            plt.close(fig)
            results[key] = {'chain_sat': [], 'chain_sur': [], 'satnames': [], 'dates_int': [], 'dates_sur': [], 'fig': fig}
            stats_dict[key] = {'rmse': None, 'bias': None, 'mean': None, 'std': None, 'r2': None}
            continue

        # Error statistics
        slope, intercept, rvalue, pvalue, std_err = stats.linregress(chain_sur, chain_sat)
        R2 = rvalue**2
        chain_error = chain_sat - chain_sur
        rmse = np.sqrt(np.mean((chain_error)**2))
        mean = np.mean(chain_error)
        std = np.std(chain_error)
        q90 = np.percentile(np.abs(chain_error), 90)
        gof = 1 - np.sum((chain_sat - chain_sur)**2) / (np.sum((np.abs(chain_sat - np.mean(chain_sur)) + np.abs(chain_sur - np.mean(chain_sur)))**2))

        # Store statistics
        stats_dict[key] = {
            'rmse': rmse,
            'bias': mean,
            #'mean': mean,
            'std': std,
            'r2': R2
        }

        # 1:1 plot
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.axis('equal')
        ax1.grid(which='major', linestyle=':', color='0.5')
        for k, sat in enumerate(list(np.unique(satnames))):
            idx = np.where([_ == sat for _ in satnames])[0]
            ax1.plot(chain_sur[idx], chain_sat[idx], 'o', ms=4, mfc=f'C{k}', mec=f'C{k}', alpha=0.7, label=sat)
        ax1.legend(loc=4)
        ax1.plot([ax1.get_xlim()[0], ax1.get_ylim()[1]], [ax1.get_xlim()[0], ax1.get_ylim()[1]], 'k--', lw=2)
        ax1.set(xlabel='Survey [m]', ylabel='Satellite [m]')
        ax1.text(0.01, 0.98, f'R2 = {R2:.2f}\nGoF = {gof:.2f}', bbox=dict(boxstyle='square', facecolor='w', alpha=1), transform=ax1.transAxes,
                 ha='left', va='top')

        # Boxplots
        ax2 = fig.add_subplot(gs[1, 1])
        data = []
        median_data = []
        n_data = []
        ax2.yaxis.grid()
        for k, sat in enumerate(list(np.unique(satnames))):
            idx = np.where([_ == sat for _ in satnames])[0]
            data.append(chain_error[idx])
            median_data.append(np.median(chain_error[idx]))
            n_data.append(len(chain_error[idx]))
        bp = ax2.boxplot(data, 0, 'k.', labels=list(np.unique(satnames)), patch_artist=True)
        for median in bp['medians']:
            median.set(color='k', linewidth=1.5)
        for j, boxes in enumerate(bp['boxes']):
            boxes.set(facecolor=f'C{j}')
            ax2.text(j + 1, median_data[j] + 1, f'{median_data[j]:.1f}', horizontalalignment='center', fontsize=12)
            ax2.text(j + 1 + 0.35, median_data[j] + 1, f'n={int(n_data[j])}', ha='center', va='center', fontsize=12, rotation='vertical')
        ax2.set(ylabel='Error [m]', ylim=settings['lims'])

        # Histogram
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.grid(which='major', linestyle=':', color='0.5')
        ax3.axvline(x=0, ls='--', lw=1.5, color='k')
        binwidth = settings['binwidth']
        bins = np.arange(min(chain_error), max(chain_error) + binwidth, binwidth)
        density = plt.hist(chain_error, bins=bins, density=True, color='0.6', edgecolor='k', alpha=0.5)
        mu, std = stats.norm.fit(chain_error)
        pval = stats.normaltest(chain_error)[1]
        xlims = ax3.get_xlim()
        x = np.linspace(xlims[0], xlims[1], 100)
        p = stats.norm.pdf(x, mu, std)
        ax3.plot(x, p, 'r-', linewidth=1)
        ax3.set(xlabel='Error [m]', ylabel='PDF', xlim=settings['lims'])
        str_stats = f' RMSE = {rmse:.1f}\n Mean = {mean:.1f}\n Std = {std:.1f}\n Q90 = {q90:.1f}'
        ax3.text(0, 0.98, str_stats, va='top', transform=ax3.transAxes)

        # Store results
        results[key] = {
            'chain_sat': chain_sat,
            'chain_sur': chain_sur,
            'satnames': satnames,
	    'dates_int' : dates_int,
	    'dates_sur': dates_sur,
            'fig': fig
        }

    return results, stats_dict


def calcular_media_mensual(dates, values):
    df = pd.DataFrame({'date': dates, 'value': values})
    df.set_index('date', inplace=True)
    monthly_means = df.groupby(df.index.to_period('M')).mean()
    monthly_means = monthly_means.to_timestamp()
    return monthly_means

# Función para calcular la tendencia y significancia
import pymannkendall as mk
def calcular_tendencia_significancia(series,alpha):
    result = mk.original_test(series,alpha)
    return result.slope, result.trend, result.h, result.p

