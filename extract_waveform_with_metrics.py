# %%

import os
import pandas as pd
import csv
import numpy as np
from phylib.utils.color import selected_cluster_color
from phylib.io.model import load_model
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.io import savemat
from scipy import signal
from scipy.signal import resample
from scipy.stats import linregress
from pathlib import Path

# based on : https://phy.readthedocs.io/en/latest/customization/
# and: https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/mean_waveforms


NAME = 'name_of_subject'
Data_folder = [R'path_to_kilosort_output_folder']
offset = 0


def main():
    model, cluster_info, cluster_df, good_clusters_id, good_spikes, spike_clusters, spike_times = read_raw()
    spikes, spikes_df = save_spikes_csv(cluster_df, good_spikes, spike_clusters, good_clusters_id, spike_times)
    extract_waveforms(model, cluster_info, good_clusters_id)
    extract_templates(model, cluster_df, cluster_info, good_clusters_id, spikes, spikes_df)
    extract_raster(spikes_df)
    Export_cluster_matlab(model, spike_times, spike_clusters, cluster_df)

def read_raw():
    # load data:
    # first argument: path to params.py
    model = load_model(Data_folder + '/params.py')
    spike_clusters = np.squeeze(np.load(Data_folder + '/spike_clusters.npy'))
    spike_times = np.squeeze(np.load(Data_folder + '/spike_times.npy'))/30000  + offset
    time_range = (spike_times[0], spike_times[-1])
    output_folder = Path(Data_folder).parent.joinpath('summary_data')

    if(os.path.isdir(output_folder) == False):
        os.makedirs(output_folder)
    # %% Save spike time as CSV file:

    # # try: #kilosort 2.5 output
    try:
        cluster_info = pd.read_csv(Data_folder + '/cluster_info.tsv', sep='\t')
    except:
        print('*********************************************************')
        print('  open data in phy and SAVE to generate cluster_info.tsv')
        print('*********************************************************')

    good_clusters = cluster_info[(cluster_info["group"] == "good") | (
        cluster_info["group"] == "mua")]["id"]

    cluster_labels = cluster_info[(cluster_info["group"] == "good") | (
        cluster_info["group"] == "mua")]["group"]

    good_clusters_id = good_clusters.to_numpy()

    good_spikes = np.isin(spike_clusters, good_clusters_id)

    cluster_df = {'cluster_id': [],'label':[], 'channel':[],"fr_mean": [], "fr_median": [], 'depth': [], 'median_activity_time': [], 'num_of_spikes': [], }
    for cluster_id in good_clusters_id:
        current_spike_time = spike_times[good_spikes][spike_clusters[good_spikes]
                                                    == cluster_id]
        cluster_df["cluster_id"].append(cluster_id)
        cluster_df["depth"].append(
            float(cluster_info["depth"][cluster_info["id"] == cluster_id]))
        cluster_df["num_of_spikes"].append(len(current_spike_time))
        channel_ids = model.get_cluster_channels(cluster_id)
        cluster_df["channel"].append(channel_ids[0])
        cluster_df["fr_mean"].append(
            float(len(current_spike_time) / (current_spike_time[-1]- current_spike_time[0]))) 
        clus_index = good_clusters[good_clusters==cluster_id].index[0]
        cluster_df["fr_median"].append(
        1 / np.median(np.diff(current_spike_time)))
        cluster_df["label"].append(cluster_labels[clus_index])
        cluster_df["median_activity_time"].append(
            np.median(current_spike_time))
        # spikes_df[spikes_df["cluster"] == cluster_id]["time"].median())

    cluster_df = pd.DataFrame(cluster_df)
    cluster_df.to_csv(output_folder.joinpath(NAME + '_clusters.csv'))

    return model, cluster_info, cluster_df, good_clusters_id, good_spikes, spike_clusters, spike_times

def save_spikes_csv(cluster_df, good_spikes, spike_clusters, good_clusters_id, spike_times):

    spikes_depth = np.zeros(spike_clusters[good_spikes].shape,'int')
    spike_good_clusters = spike_clusters[good_spikes]

    for cluster_id in good_clusters_id:
        spikes_depth[spike_good_clusters == cluster_id] = int(cluster_df[cluster_df["cluster_id"] == cluster_id]["depth"])
    # spikes_depth = [int(cluster_df[cluster_df["cluster_id"] == cluster_id]["depth"])
    #                 for cluster_id in spike_clusters[good_spikes]]

    spikes = {'cluster_id': spike_clusters[good_spikes],
              'time': spike_times[good_spikes],
              'depth': spikes_depth}

    spikes_df = pd.DataFrame(data=spikes)
    output_folder = Path(Data_folder).parent.joinpath('summary_data')
    spikes_df.to_csv(output_folder.joinpath(NAME + '_spikes.csv'))
    return spikes, spikes_df

def extract_waveforms(model, cluster_info, good_clusters_id):
    output_folder = Path(Data_folder).parent.joinpath('summary_data')
    output_folder = Path(Data_folder).parent.joinpath('summary_data')

    output_fname = output_folder.joinpath(NAME + '_cluster_summary.pdf')
    with PdfPages(output_fname) as pdf:
        # First, we load the TemplateModel.
        # model = load_model(sys.argv[1])  # first argument: path to params.py
        # int(sys.argv[2])  # second argument: cluster index
        cluster_id = int(1)

        # We obtain the cluster id from the command-line arguments.
        clust_per_page = 5
        num_of_ch_to_show = 4
        cluster_order = list(range(0, len(good_clusters_id), clust_per_page))
        for i in cluster_order:  # list([0,5]):
            index = 0
            f, axes = plt.subplots(
                figsize=(20, 5*3), nrows=clust_per_page, ncols=num_of_ch_to_show, sharey=True)

            for cluster_id in good_clusters_id[i:min(i+5, len(good_clusters_id))]:
                # We get the waveforms of the cluster.
                waveforms = model.get_cluster_spike_waveforms(cluster_id)
                n_spikes, n_samples, n_channels_loc = waveforms.shape

                # We get the channel ids where the waveforms are located.
                channel_ids = model.get_cluster_channels(cluster_id)

                # We plot the waveforms on the first four channels.
                for ch in range(min(num_of_ch_to_show, n_channels_loc)):
                    axes[index, ch].plot(
                        waveforms[::max(int(n_spikes/50), 1), :, ch].T, c=selected_cluster_color(0, .5))
                    axes[index, ch].plot(
                        np.median(waveforms[:, :, ch], 0), c='k')
                    axes[index, ch].set_title("channel %d" % channel_ids[ch])

                    if ch == 0:
                        # fr = n_spikes/720
                        fr = cluster_info["fr"][cluster_info["id"] == cluster_id]
                        axes[index, ch].set_ylabel(
                            'clus#{} FR = {:.2f}'.format(cluster_id, float(fr)), fontsize=14)

                index += 1
            # pdf.savefig(bbox_inches = 'tight')
            pdf.savefig()
            plt.close()

def extract_templates(model, cluster_df, cluster_info, good_clusters_id, spikes, spikes_df):
    output_folder = Path(Data_folder).parent.joinpath('summary_data')
    # waveforms = model.get_cluster_spike_waveforms(1)
    template = model.get_template_waveforms(5)
    n_samples, n_channels_loc = template.shape
    clust_per_page = 6
    model.get_template_channels(1)
    output_fname = output_folder.joinpath(NAME + '_template_summary.pdf')

    # clus = np.unique(spikes_df["cluster"])
    # compute median location of spikes in time (activity center):

    # sort according to activity centers?:
    ordered_list_time = [cluster_df["cluster_id"][i]
                         for i in np.argsort(cluster_df["median_activity_time"])]
    ordered_list_depth = [cluster_df["cluster_id"][i]
                          for i in np.argsort(cluster_df["depth"])]
    ordered_list_fr = [cluster_df["cluster_id"][i]
                       for i in np.argsort(cluster_df["num_of_spikes"])]

    with PdfPages(output_fname) as pdf:
        sns.set(rc={'figure.figsize': (36, 18)}, font_scale=5)
        sns.boxplot(x="cluster_id", y="time", data=spikes_df, order=ordered_list_time).set_title(
            "Spikes box plot, sorted by median time of activity")
        pdf.savefig()
        plt.close()

        sns.set(rc={'figure.figsize': (36, 18)}, font_scale=5)
        sns.boxplot(x="cluster_id", y="time", data=spikes_df, order=ordered_list_depth).set_title(
            "Spikes box plot, sorted by depth")
        pdf.savefig()
        plt.close()

        sns.set(rc={'figure.figsize': (36, 16)}, font_scale=5)
        sns.boxplot(x="cluster_id", y="time", data=spikes_df, order=ordered_list_fr).set_title(
            "Spikes box plot, sorted by num of spikes")
        pdf.savefig()
        plt.close()

        cluster_df.plot.scatter(x='depth',
                                y='fr_mean',
                                s=300,
                                c='median_activity_time',
                                colormap='viridis',
                                title="Transient firing rate vs. Depth",
                                xlabel="Depth (um)",
                                ylabel="Firing Rate (Hz)",
                                ylim=(0, 15))
        pdf.savefig()
        plt.close()

        # First, we load the TemplateModel.
        # model = load_model(sys.argv[1])  # first argument: path to params.py
        # int(sys.argv[2])  # second argument: cluster index
        cluster_id = int(1)

        # We obtain the cluster id from the command-line arguments.
        cluster_order = list(range(0, len(good_clusters_id), clust_per_page))

        for i in cluster_order:  # list([0,5]):
            index = 0
            f, axes = plt.subplots(figsize=(36, 18), nrows=1,
                                   ncols=clust_per_page, sharey=True)

            for cluster_id in good_clusters_id[i:min(i+clust_per_page, len(good_clusters_id))]:
                # We get the waveforms of the cluster.
                template = np.median(model.get_cluster_spike_waveforms(cluster_id), axis=0)
                # template = model.get_template(cluster_id).template
                # channels = model.get_template_channels(cluster_id)
                channels = model.get_cluster_channels(cluster_id)

                ch_order = np.argsort(channels)

                for ch in range(n_channels_loc):
                    axes[index].axis('off')
                    axes[index].plot(50*ch + template[:, ch_order[ch]])
                    depth = int(cluster_info["depth"]
                                [cluster_info["id"] == cluster_id])
                    fr = float(cluster_info["fr"]
                               [cluster_info["id"] == cluster_id])
                    fr_mean = 1 / \
                        np.mean(
                            np.diff(spikes["time"][spikes["cluster_id"] == cluster_id]))
                    fr_median = 1 / \
                        np.median(
                            np.diff(spikes["time"][spikes["cluster_id"] == cluster_id]))
                    axes[index].set_title('cluster {} \n Depth = {} um \n FRdata = {:.2f}Hz \n FRmean = {:.2f}Hz \n FRtransient = {:.2f}Hz'.format(
                        cluster_id, depth, fr, fr_mean, fr_median), fontsize=18)

                index += 1
            pdf.savefig()
            plt.close()

def extract_raster(spikes_df):
    output_folder = Path(Data_folder).parent.joinpath('summary_data')
    output_fname = output_folder.joinpath(NAME + '_spikes_scatter.pdf')
    output_fname_png = output_folder.joinpath(NAME + '_spikes_scatter.png')

    with PdfPages(output_fname) as pdf:
        tt = (spikes_df["time"].min(), spikes_df["time"].max())

        sns.set(rc={'figure.figsize': (36, 18)}, font_scale=1)
        g = sns.JointGrid(data=spikes_df, x="time",
                          y="depth", marginal_ticks=True)
        # g.plot_joint(sns.scatterplot, size = 0.01*np.ones(len(spikes_df)))
        g.plot_joint(plt.scatter, s=0.1*np.ones(len(spikes_df)), c='k')

        g.plot_marginals(sns.histplot, element="poly", color="#03012d", bins=int(tt[1]-tt[0]),
                         weights=np.ones(len(spikes_df))/int(tt[1]-tt[0]))
        g.set_axis_labels(xlabel='Time (sec)', ylabel='Depth (uM)')
        g.savefig(output_fname_png, dpi=600)
        pdf.savefig()
        plt.close()

def calculate_snr(W):
    
    """
    Calculate SNR of spike waveforms.

    Converted from Matlab by Xiaoxuan Jia

    ref: (Nordhausen et al., 1996; Suner et al., 2005)

    Input:
    -------
    W : array of N waveforms (N x samples)

    Output:
    snr : signal-to-noise ratio for unit (scalar)

    """

    W_bar = np.nanmean(W, axis=0)
    A = np.max(W_bar) - np.min(W_bar)
    e = W - np.tile(W_bar, (np.shape(W)[0], 1))
    snr = A/(2*np.nanstd(e.flatten()))

    return snr

def calculate_waveform_duration(waveform, timestamps):
    
    """ 
    Duration (in seconds) between peak and trough

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)

    Outputs:
    --------
    duration : waveform duration in milliseconds

    """

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform)

    # to avoid detecting peak before trough
    if waveform[peak_idx] > np.abs(waveform[trough_idx]):
        duration =  timestamps[peak_idx:][np.where(waveform[peak_idx:]==np.min(waveform[peak_idx:]))[0][0]] - timestamps[peak_idx] 
    else:
        duration =  timestamps[trough_idx:][np.where(waveform[trough_idx:]==np.max(waveform[trough_idx:]))[0][0]] - timestamps[trough_idx] 

    return duration * 1e3
  
def calculate_waveform_halfwidth(waveform, timestamps):
    
    """ 
    Spike width (in seconds) at half max amplitude

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)

    Outputs:
    --------
    halfwidth : waveform halfwidth in milliseconds

    """

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform)

    try:
        if waveform[peak_idx] > np.abs(waveform[trough_idx]):
            threshold = waveform[peak_idx] * 0.5
            thresh_crossing_1 = np.min(
                np.where(waveform[:peak_idx] > threshold)[0])
            thresh_crossing_2 = np.min(
                np.where(waveform[peak_idx:] < threshold)[0]) + peak_idx
        else:
            threshold = waveform[trough_idx] * 0.5
            thresh_crossing_1 = np.min(
                np.where(waveform[:trough_idx] < threshold)[0])
            thresh_crossing_2 = np.min(
                np.where(waveform[trough_idx:] > threshold)[0]) + trough_idx

        halfwidth = (timestamps[thresh_crossing_2] - timestamps[thresh_crossing_1]) 

    except ValueError:

        halfwidth = np.nan

    return halfwidth * 1e3

def calculate_waveform_PT_ratio(waveform):

    """ 
    Peak-to-trough ratio of 1D waveform

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)

    Outputs:
    --------
    PT_ratio : waveform peak-to-trough ratio

    """

    trough_idx = np.argmin(waveform)

    peak_idx = np.argmax(waveform)

    PT_ratio = np.abs(waveform[peak_idx] / waveform[trough_idx])

    return PT_ratio

def calculate_waveform_repolarization_slope(waveform, timestamps, window=20):
    
    """ 
    Spike repolarization slope (after maximum deflection point)

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)
    window : int
        Window (in samples) for linear regression

    Outputs:
    --------
    repolarization_slope : slope of return to baseline (V / s)

    """

    max_point = np.argmax(np.abs(waveform))

    waveform = - waveform * (np.sign(waveform[max_point])) # invert if we're using the peak

    repolarization_slope = linregress(timestamps[max_point:max_point+window], waveform[max_point:max_point+window])[0]

    return repolarization_slope * 1e-6

def calculate_waveform_recovery_slope(waveform, timestamps, window=20):

    """ 
    Spike recovery slope (after repolarization)

    Inputs:
    ------
    waveform : numpy.ndarray (N samples)
    timestamps : numpy.ndarray (N samples)
    window : int
        Window (in samples) for linear regression

    Outputs:
    --------
    recovery_slope : slope of recovery period (V / s)

    """

    max_point = np.argmax(np.abs(waveform))

    waveform = - waveform * (np.sign(waveform[max_point])) # invert if we're using the peak

    peak_idx = np.argmax(waveform[max_point:]) + max_point

    recovery_slope = linregress(timestamps[peak_idx:peak_idx+window], waveform[peak_idx:peak_idx+window])[0]

    return recovery_slope * 1e-6

def calculate_2D_features(waveform, timestamps, peak_channel, spread_threshold = 0.12, site_range=16, site_spacing=10e-6):
    
    """ 
    Compute features of 2D waveform (channels x samples)

    Inputs:
    ------
    waveform : numpy.ndarray (N channels x M samples)
    timestamps : numpy.ndarray (M samples)
    peak_channel : int
    spread_threshold : float
    site_range: int
    site_spacing : float

    Outputs:
    --------
    amplitude : uV
    spread : um
    velocity_above : s / m
    velocity_below : s / m

    """

    assert site_range % 2 == 0 # must be even

    sites_to_sample = np.arange(-site_range, site_range+1, 2) + peak_channel

    sites_to_sample = sites_to_sample[(sites_to_sample > 0) * (sites_to_sample < waveform.shape[0])]

    wv = waveform[sites_to_sample, :]

    #smoothed_waveform = np.zeros((wv.shape[0]-1,wv.shape[1]))
    #for i in range(wv.shape[0]-1):
    #    smoothed_waveform[i,:] = np.mean(wv[i:i+2,:],0)

    trough_idx = np.argmin(wv, 1)
    trough_amplitude = np.min(wv, 1)

    peak_idx = np.argmax(wv, 1)
    peak_amplitude = np.max(wv, 1)

    duration = np.abs(timestamps[peak_idx] - timestamps[trough_idx])

    overall_amplitude = peak_amplitude - trough_amplitude
    amplitude = np.max(overall_amplitude)
    max_chan = np.argmax(overall_amplitude)

    points_above_thresh = np.where(overall_amplitude > (amplitude * spread_threshold))[0]
    
    if len(points_above_thresh) > 1:
        points_above_thresh = points_above_thresh[isnot_outlier(points_above_thresh)]

    spread = len(points_above_thresh) * site_spacing * 1e6

    channels = sites_to_sample - peak_channel
    channels = channels[points_above_thresh]

    trough_times = timestamps[trough_idx] - timestamps[trough_idx[max_chan]]
    trough_times = trough_times[points_above_thresh]

    velocity_above, velocity_below = get_velocity(channels, trough_times, site_spacing)
 
    return amplitude, spread, velocity_above, velocity_below

def get_velocity(channels, times, distance_between_channels = 10e-6):
    
    """
    Calculate slope of trough time above and below soma.

    Inputs:
    -------
    channels : np.ndarray
        Channel index relative to soma
    times : np.ndarray
        Trough time relative to peak channel
    distance_between_channels : float
        Distance between channels (m)

    Outputs:
    --------
    velocity_above : float
        Inverse of velocity of spike propagation above the soma (s / m)
    velocity_below : float
        Inverse of velocity of spike propagation below the soma (s / m)

    """

    above_soma = channels >= 0
    below_soma = channels <= 0

    if np.sum(above_soma) > 1:
        slope_above, intercept, r_value, p_value, std_err = linregress(channels[above_soma], times[above_soma])
        velocity_above = slope_above / distance_between_channels
    else:
        velocity_above = np.nan

    if np.sum(below_soma) > 1:
        slope_below, intercept, r_value, p_value, std_err = linregress(channels[below_soma], times[below_soma])
        velocity_below = slope_below / distance_between_channels
    else:
        velocity_below = np.nan

    return velocity_above, velocity_below

def isnot_outlier(points, thresh=1.5):

    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """

    if len(points.shape) == 1:
        points = points[:,None]

    median = np.median(points, axis=0)
    
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score <= thresh

def Export_cluster_matlab(model, spike_times, spike_clusters, cluster_df):
    wave_matric = pd.DataFrame()
    output_folder = Path(Data_folder).parent.joinpath('Matlab_clusters')
    if(os.path.isdir(output_folder) == False):
        os.makedirs(output_folder)

    max_shift = 20
    data_to_save = cluster_df.iloc[0]
    for i, data_to_save in cluster_df.iterrows():
        data_to_save = data_to_save.to_dict()
        clust_id = int(data_to_save["cluster_id"])
        data_to_save["channels"] = model.get_cluster_channels(clust_id)
        data_to_save["spikes"] = spike_times[spike_clusters==clust_id] 

        waveforms = model.get_cluster_spike_waveforms(clust_id)
        new_waveforms = np.zeros(waveforms.shape)

        mean_waveform =( np.mean(waveforms[:,:,0],0), np.mean(waveforms[:,:,1],0))

        w0 = waveforms[0,:,0]
        w1 = waveforms[0,:,1]
        for i in range(len(waveforms[:,0,0])):
            c0 = signal.correlate(mean_waveform[0], waveforms[i,:,0])
            temp0 = max_shift - 1 - np.argmax(c0[len(w0)-max_shift: len(w0)+max_shift])

            c1 = signal.correlate(mean_waveform[1], waveforms[i,:,1])
            temp1 = max_shift - 1 - np.argmax(c0[len(w1)-max_shift: len(w1)+max_shift])

            shift = (np.min([temp0,temp1]))
            t = range(82) + shift
            t = [x for x in t if x>=0 and x<82]
            for w in range(waveforms.shape[2]):
                new_waveforms[i,t-shift,w] = waveforms[i,t,w]

        # sort first 3 channels according to average power: 
        ch_order = np.argsort(np.sum(np.power(np.mean(new_waveforms[:,:,:3],0),2),0))[::-1]
        channels = data_to_save["channels"]
        channels[range(len(ch_order))] = channels[ch_order]
        peak_channel = channels[0]

        waveforms = np.zeros(new_waveforms.shape)
        # waveforms = new_waveforms[:,:,channels]
        waveforms[:,:,channels] = new_waveforms # sort channel order ,waveform channel order is 1:384

        data_to_save["channels"] = channels[0:11]
        data_to_save["peak_channel"] = peak_channel
        data_to_save["waveforms"] = new_waveforms[:,:,0:11]
        data_to_save["templates"] = np.mean(new_waveforms[:,:,0:12],0)
        metrics = waveform_metrics(waveforms, peak_channel, clust_id)
        data_to_save.update(metrics)
        wave_matric = wave_matric.append(pd.DataFrame(metrics))
        file_name =output_folder.joinpath(NAME + '_cluster_' + str(clust_id) + '.mat')
        savemat(file_name, data_to_save)
    output_folder = Path(Data_folder).parent.joinpath('summary_data')
    wave_matric.to_csv(output_folder .joinpath(NAME + 'waveform_metrics.csv'))

def waveform_metrics(new_waveforms, peak_channel, clust_id):
    sample_rate = 30000
    upsampling_factor = 1

    waveforms =  np.swapaxes(new_waveforms, 1, 2)
    # ch_order = np.argsort(channels)

    mean_2D_waveform = np.squeeze(np.nanmean(waveforms, 0))
    # local_peak = np.argmin(np.abs(channels[ch_order] - channels[0] ))

    snr = calculate_snr(waveforms[:, 0, :])

    num_samples = waveforms.shape[2]
    new_sample_count = int(num_samples * upsampling_factor)

    mean_1D_waveform = resample(np.mean(waveforms[:, peak_channel, :],0), new_sample_count)
    # plt.plot(mean_1D_waveform)
    # plt.show()
    timestamps = np.linspace(0, num_samples / sample_rate, new_sample_count)

    duration = calculate_waveform_duration(mean_1D_waveform, timestamps)
    halfwidth = calculate_waveform_halfwidth(mean_1D_waveform, timestamps)
    PT_ratio = calculate_waveform_PT_ratio(mean_1D_waveform)
    repolarization_slope = calculate_waveform_repolarization_slope(mean_1D_waveform, timestamps, window=20*upsampling_factor)
    recovery_slope = calculate_waveform_recovery_slope(mean_1D_waveform, timestamps, window=20*upsampling_factor)
    
    amplitude, spread, velocity_above, velocity_below = calculate_2D_features(mean_2D_waveform, timestamps, peak_channel)
    
    # data = [[clust_id, snr, duration, halfwidth, PT_ratio, repolarization_slope,
    #             recovery_slope]]

    # metrics = pd.DataFrame(data,
    #         columns=['cluster_id', 'snr', 'spike_duration', 'halfwidth',
    #                                     'PT_ratio', 'repolarization_slope', 'recovery_slope'])
    metrics = {'cluster_id':[clust_id], 
                'snr':[snr], 
                'spike_duration':[duration], 
                'halfwidth':[halfwidth],
                'PT_ratio':[PT_ratio], 
                'repolarization_slope':[repolarization_slope], 
                'recovery_slope':[recovery_slope],
                'amplitude':[amplitude],
                'spread':[spread],
                'velocity_above':[velocity_above],
                'velocity_below':[velocity_below]
                }
    return metrics

main()
