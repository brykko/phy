# -*- coding: utf-8 -*-

"""SpatialView"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Base imports
import logging
import pdb
from PyQt4.QtCore import pyqtRemoveInputHook
import numpy as np

from phy.utils._color import _colormap
from phy.utils import Bunch
from .base import ManualClusteringView
from phy.plot import NDC

# RG imports
from bisect import bisect_left
import math
import scipy.io as sio
import scipy.ndimage as sim
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SpatialView(ManualClusteringView):

    # Some class variables
    n_rate_map_bins = 100
    n_rate_map_bins_sigma = 4
    n_rate_map_contours = 10
    rate_map_contour_mode = "zero"
    n_hd_bins = 180
    n_hd_bins_sigma = 3
    sec_smooth_pos = 0
    sec_smooth_hd = 0.2
    sec_smooth_speed = 0.5
    speed_threshold = 0.025
    speed_threshold_mode = "above"

    default_shortcuts = {
        'go_left': 'alt+left',
        'go_right': 'alt+right',
    }

    def __init__(self,
                 spike_samples=None,
                 spike_clusters=None,
                 sample_rate=None,
                 tracking_data=None,
                 time_ranges=None,
                 **kwargs):

        assert sample_rate > 0

        # Initialize the view.
        super(SpatialView, self).__init__(
            layout='grid',
            shape=(2, 2),
            **kwargs)

        self.spike_clusters = spike_clusters
        self.sample_rate = float(sample_rate)
        self.spike_samples = np.asarray(spike_samples)

        if time_ranges is None:
            self.time_ranges = ((0, self.spike_samples[-1]/self.sample_rate), )
        else:
            self.time_ranges = time_ranges

        # Apply the timerange restriction to the data if necessary
        self._do_plot = True if tracking_data is not None else False

        if tracking_data is not None:
            logger.debug("%u tracking points received", tracking_data.shape[0])
            self.tracking_data = tracking_data
            self.tracking_data_smoothed = None
            self._smooth_tracking_data()
            self._update_status()
            self._apply_tracking_validity()
        else:
            logger.debug("No tracking data received, SpatialView will not draw.")

    def _smooth_tracking_data(self):
        d = self.tracking_data
        d_sm = d
        tracking_t = d[:, 0] / self.sample_rate
        tracking_fs = 1 / np.mean(np.diff(tracking_t))
        sigma_pos = math.ceil(self.sec_smooth_pos * tracking_fs)
        sigma_hd = math.ceil(self.sec_smooth_hd * tracking_fs)
        sigma_speed = math.ceil(self.sec_smooth_speed * tracking_fs)

        # position
        if sigma_pos > 0:
            for i in range(2):
                d_sm[:, 1+i] = gaussian_filter1d(d[:, 1+i], sigma_pos, truncate=2)

        # azimuth angle
        if sigma_hd > 0:
            hd = d[:, 3]
            data_filt = [gaussian_filter1d([np.sin, np.cos][i](hd), sigma_hd, truncate=2)
                         for i in range(2)]
            d_sm[:, 3] = np.mod(np.arctan2(data_filt[0], data_filt[1]), 2*math.pi)

        self.tracking_data_smoothed = d_sm

        # speed
        dpos = list()
        for i in range(2):
            if sigma_speed == 0:
                dtmp = gaussian_filter1d(d[:, 1+i], sigma_speed, truncate=2)
            else:
                dtmp = d[:, 1+i]
            dpos.append(np.diff(dtmp))
        self.speed = np.hstack((0, np.hypot(dpos[0], dpos[1]) * tracking_fs))

    def _apply_tracking_validity(self):
        """
        Apply the current 'timerange' and 'speed_threshold' restrictions to the
        tracking data
        """
        d = self.tracking_data
        spike_samples = self.spike_samples

        valid_t_spikes = np.zeros(shape=spike_samples.shape, dtype="bool")
        valid_t_tracking = np.zeros(shape=d.shape[0], dtype="bool")
        for trng in self.time_ranges:
            srng = [val * self.sample_rate for val in trng]
            valid_t_spikes |= (spike_samples > srng[0]) & (spike_samples < srng[1])
            valid_t_tracking |= (d[:, 0] > srng[0]) & (d[:, 0] < srng[1])

        inds_spike_tracking = _binary_search(d[:, 0], spike_samples)
        inds_spike_tracking[~valid_t_spikes] = 0
        spike_speed = self.speed[inds_spike_tracking]

        if self.speed_threshold_mode == "above":
            valid_speed = self.speed >= self.speed_threshold
            valid_speed_spikes = spike_speed >= self.speed_threshold
        else:
            valid_speed = self.speed <= self.speed_threshold
            valid_speed_spikes = spike_speed <= self.speed_threshold

        self.valid_spikes = valid_t_spikes & valid_speed_spikes
        self.valid_tracking_time = valid_t_tracking
        self.valid_tracking = valid_t_tracking & valid_speed

        if any(self.valid_tracking):
            self._calc_occupancy()
            self._do_plot = True
        else:
            logger.warning('No valid tracking points exist. SpatialView plots will not be drawn')
            self._do_plot = False

    def on_select(self, cluster_ids=None):
        super(SpatialView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters > 0 and self._do_plot:
            with self.building():
                for c, id in enumerate(cluster_ids):
                    self._make_plots(c, id)

    def _make_plots(self, clu_selection_idx, cluster_id):
        """
        Generate all plots for one cluster
        :param clu_selection_idx: index of cluster in the current selection
        :param cluster_id: ID of cluster
        :return:
        """
        if clu_selection_idx is not None:
            color = tuple(_colormap(clu_selection_idx))
            color_transp = color + (0.5,)
            color_solid = color + (1.0,)
        else:
            return
        assert len(color) == 3
        
        pos = self.tracking_data
        x = pos[self.valid_tracking, 1]
        y = pos[self.valid_tracking, 2]

        # Get indices of all spikes for the current cluster
        spikes_in_clu = np.isin(self.spike_clusters, cluster_id)
        spike_samples = self.spike_samples[spikes_in_clu & self.valid_spikes]

        inds_spike_tracking = _binary_search(pos[:, 0], spike_samples)
        valid_spikes = inds_spike_tracking >= 0
        inds_spike_tracking = inds_spike_tracking[valid_spikes]
        hd_tuning_curve, spike_hd = self._hd_tuning_curve(inds_spike_tracking)

        # Find range of X/Y tracking data
        min_x = np.min(x)
        min_y = np.min(y)
        max_x = np.max(x)
        max_y = np.max(y)
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        rng_x = max_x - min_x
        rng_y = max_y - min_y
        max_rng = max(rng_x, rng_y)
        hw = max_rng / 2
        data_bounds = (mid_x-hw, mid_y-hw, mid_x+hw, mid_y+hw)

        # Plot path first time only
        if clu_selection_idx == 0:
            # Both normal and hd-coded
            for i in [0, 1]:
                self[0, i].uplot(
                    x=pos[self.valid_tracking_time, 1].reshape((1,-1)),
                    y=pos[self.valid_tracking_time, 2].reshape((1, -1)),
                    color=(1, 1, 1, 0.2),
                    data_bounds=data_bounds)

        # Spike locations
        self[0, 0].scatter(
            x=pos[inds_spike_tracking, 1],
            y=pos[inds_spike_tracking, 2],
            color=color_transp,
            size=2,
            data_bounds=data_bounds)

        # Spike locations (HD-color-coded)
        spike_colors = _vector_to_rgb(spike_hd)
        self[0, 1].scatter(
            x=pos[inds_spike_tracking, 1],
            y=pos[inds_spike_tracking, 2],
            color=spike_colors,
            size=2,
            data_bounds=data_bounds)

        # Rate map contour
        contours = self._2d_rate_map_contours(inds_spike_tracking)
        v = np.linspace(0, 1, len(contours))
        for i, contour in enumerate(contours):
            contour_color = tuple(color) + (v[i],)
            for line in contour:
                x = line[:, 1]
                y = line[:, 0]
                self[1, 0].plot(
                    x=x,
                    y=y,
                    color=contour_color,
                    data_bounds=data_bounds)

        # HD plot
        rho = 1.1
        (pol_x, pol_y) = _pol2cart(rho, np.linspace(0, 2 * math.pi, 1000))
        min_x = np.min(pol_x)
        min_y = np.min(pol_y)
        max_x = np.max(pol_x)
        max_y = np.max(pol_y)
        data_bounds = (min_x, min_y, max_x, max_y)

        # Plot axes first time only
        if clu_selection_idx == 0:
            self[1, 1].uplot(
                    x=pol_x,
                    y=pol_y,
                    color=(1, 1, 1, 0.5),
                    data_bounds=data_bounds)

            self[1, 1].uplot(
                x=np.array([min_x, max_x]),
                y=np.array([0, 0]) + (min_y + max_y)/2,
                color=(1, 1, 1, 0.3),
                data_bounds=data_bounds)

            self[1, 1].uplot(
                y=np.array([min_y, max_y]),
                x=np.array([0, 0]) + (min_x + max_x)/2,
                color=(1, 1, 1, 0.3),
                data_bounds=data_bounds)

        bin_size_hd = self.bins['hd'][1] - self.bins['hd'][0]
        bin_centres_hd = self.bins['hd'][:-1] + bin_size_hd/2
        (x, y) = _pol2cart(hd_tuning_curve, bin_centres_hd)

        # HD tuning curve
        self[1, 1].plot(
                x=x,
                y=y,
                color=color_solid,
                data_bounds=data_bounds)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(SpatialView, self).attach(gui)
        self.actions.add(self.set_speed_threshold)
        self.actions.add(self.toggle_speed_threshold_mode)
        self.actions.add(self.set_time_range)
        self.actions.separator()
        self.actions.add(self.set_rate_map_contour_count)
        self.actions.add(self.toggle_rate_map_contour_minimum)
        self.actions.add(self.set_rate_map_n_smooth)

    @property
    def state(self):
        return Bunch(speed_threshold=self.speed_threshold,
                     speed_threshold_mode=self.speed_threshold_mode,
                     time_range=self.time_ranges,
                     n_rate_map_contours=self.n_rate_map_contours,
                     rate_map_contour_mode=self.rate_map_contour_mode)

    def set_speed_threshold(self, speed_threshold):
        """
        Set the threshold value for locomotion speed (in m/s).

        Example: '0.05'
        """
        self.speed_threshold = float(speed_threshold)
        self._apply_tracking_validity()
        self._update_status()
        self.on_select()

    def toggle_speed_threshold_mode(self):
        """
        Toggle between selecting spikes above and below the speed threshold.
        """
        self.speed_threshold_mode = {
            "above": "below",
            "below": "above"}[self.speed_threshold_mode]
        self._apply_tracking_validity()
        self._update_status()
        self.on_select()

    def set_time_range(self, time_range, *args):
        """
        Define the recording time range for which plots will be generated. Time
        must be in seconds relative to the start of ephys data acquisition, and
        must be specified as a list or tuple. By default, all available tracking
        data will be plotted. Reset the default state by typing 'all' as the
        time-range value.

        A time range must be specified as two digits with an underscore '_'
        between them.

        Example: '0_3600' selects tracking data from the first hour.

        Multiple time ranges may be specified as a comma-separated list.

        Example '0_100, 500_600, 1000_2000' specifies three time ranges.
        """
        # TODO: force the action callback to return string type

        # Check if first arg is a list of strings
        if type(time_range) is not str:
            time_range = time_range[0]

        strs = list()
        strs.append(time_range)
        for s in args:
            if type(s) == str:
                strs.append(s)
            else:
                strs.append(s[0])

        if time_range == "all":
            time_ranges = ((0, float(self.spike_samples[-1]/self.sample_rate)), )
        else:
            time_ranges = [[float(s.split("_")[n].strip(",")) for n in range(2)] for s in strs]
        self.time_ranges = time_ranges
        self._apply_tracking_validity()
        self._update_status()
        self.on_select()

    def set_rate_map_contour_count(self, contour_count):
        """
        Define how many contour levels are plotted. The default number is 10.
        """
        self.n_rate_map_contours = int(contour_count)
        self._update_status()
        self.on_select()

    def set_rate_map_n_smooth(self, n_smooth):
        """
        Define the standard deviation of the gaussian smoothing kernel.
        Units are in bins. The default value is 2.
        """
        assert n_smooth >= 0
        self.n_rate_map_bins_sigma = n_smooth
        self._update_status()
        self.on_select()

    def toggle_rate_map_contour_minimum(self):
        """
        Toggle the method for determining the minimum rate map contour level.
        The two modes are 'zero', where contour levels begin at zero, and
        'minimum', where contours begin at the rate map's minimum value.
        """
        self.rate_map_contour_mode = {
            "zero": "minimum",
            "minimum": "zero"}[self.rate_map_contour_mode]
        self._update_status()
        self.on_select()

    def _update_status(self):
        s = ""
        for i, t in enumerate(self.time_ranges):
            s += "{:.2f}-{:.2f}".format(t[0], t[1])
            if i < len(self.time_ranges)-1:
                s += ", "
        str_timerange = s

        if self.speed_threshold_mode == "above":
            str_ineq = ">="
        else:
            str_ineq = "<="

        self.set_status(
            "t={}, speed{}{:.3f} m/s, n contours={}, min contour='{}', smooth={}"
            .format(
                str_timerange,
                str_ineq,
                self.speed_threshold,
                self.n_rate_map_contours,
                self.rate_map_contour_mode,
                self.n_rate_map_bins_sigma))

    def _hd_tuning_curve(self, spike_tracking_inds):
        # Get the HD for every spike
        hd = self.tracking_data[:, 3]
        spike_hd = hd[spike_tracking_inds]

        # Make the histogram
        tmp = np.histogram(a=spike_hd, bins=(self.bins['hd']))
        hist_spike = tmp[0]

        # Normalize by the occupancy hist to get tuning curve
        tcurve = hist_spike / self.hd_occupancy_hist
        bad_bins = np.isinf(tcurve) | np.isnan(tcurve)
        tcurve[bad_bins] = 0

        # Gaussian smooth the tuning curve
        tcurve = sim.filters.gaussian_filter(
                tcurve,
                sigma=self.n_hd_bins_sigma,
                mode='wrap')
        tcurve /= np.max(tcurve)
        return tcurve, spike_hd
     
    def _calc_occupancy(self):
        pos = self.tracking_data
        v = self.valid_tracking
        x = pos[v, 1]
        y = pos[v, 2]

        # Make the bin vectors for x and y
        self.bins = dict()
        vars = {'x': x, 'y': y}
        for key, val in vars.items():
            min = np.percentile(val, 0.1)
            max = np.percentile(val, 99.9)
            mid = (max + min) / 2
            range = max - min
            self.bins[key] = np.linspace(mid-range*0.6, mid+range*0.6, self.n_rate_map_bins)
           
        self.pos_occupancy_hist = np.histogram2d(
            x, y, bins=(self.bins['x'], self.bins['y']))[0]

        # Make HD occupancy histogram
        self.bins['hd'] = np.linspace(0, 2*math.pi, self.n_hd_bins)
        self.hd_occupancy_hist = np.histogram(
            pos[v, 3], bins=(self.bins['hd']))[0]

    def _2d_rate_map_contours(self, spike_tracking_inds):
        # Generate contour vertices for 2d position rate map
        pos = self.tracking_data
        bx = self.bins['x']
        by = self.bins['y']

        spike_counts = np.histogram2d(
            pos[spike_tracking_inds, 1],
            pos[spike_tracking_inds, 2],
            bins=(bx, by))[0]

        occ = self.pos_occupancy_hist
        rate_map = spike_counts / occ
        rate_map[occ == 0] = np.nanmean(rate_map)
        rate_map = sim.filters.gaussian_filter(
            rate_map,
            sigma=self.n_rate_map_bins_sigma)

        rate_map /= rate_map.mean()
        bxc = _edges_to_centers(bx)
        byc = _edges_to_centers(by)

        if self.rate_map_contour_mode == "zero":
            min_val = 0
        else:
            min_val = np.min(rate_map)

        v = np.linspace(min_val, np.max(rate_map), self.n_rate_map_contours+2)
        v = v[1:-1]
        contour = plt.contour(bxc, byc, rate_map, v)
        coords = list()
        for c in contour.collections:
            coords.append([p.vertices for p in c.get_paths()])
        return coords


# -----------------------------------------------------------------------------
# Internal helper functions
# -----------------------------------------------------------------------------

def _cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def _pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def _vector_to_rgb(a):
    h = a[:, None]/(2*np.pi)
    s = np.ones(np.shape(h))
    v = np.ones(np.shape(h))
    l = np.ones(np.shape(h))
    hsv = np.hstack((h, s, v))
    rgb = hsv_to_rgb(hsv)
    return np.hstack((rgb, l)).astype('float32')


def _binary_search(a, x, lo=0, hi=None):
    idx = np.empty((len(x)), dtype='int32')
    for i in range(len(x)):
        hi = hi if hi is not None else len(a)   # hi defaults to len(a)
        tmp = bisect_left(a, x[i], lo, hi)      # find insertion position
        idx_tmp = (tmp if tmp != hi else -1)     # don't walk off the end
        idx[i] = idx_tmp
    return idx


def _edges_to_centers(x):
    assert(x.size >= 2)
    dx = x[1]-x[0]
    return x[0:-1] + dx/2
