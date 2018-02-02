# -*- coding: utf-8 -*-

"""Spatial view."""

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

logger = logging.getLogger(__name__)

# RG imports
from bisect import bisect_left
import math
import scipy.io as sio
import scipy.ndimage as sim
from matplotlib.colors import hsv_to_rgb

# -----------------------------------------------------------------------------
# Spatial view
# -----------------------------------------------------------------------------


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def array_to_rgb(a):
    h = a/(2*np.pi)
    s = np.ones(np.shape(h))
    v = np.ones(np.shape(h))
    l = np.ones(np.shape(h))
    hsv = np.hstack((h, s, v))
    rgb = hsv_to_rgb(hsv)
    return np.hstack((rgb, l)).astype('float32')

    
def binary_search(a, x, lo=0, hi=None):
    
    idx = np.empty((len(x), 1), dtype='int32')
    for i in range(len(x)):
        hi = hi if hi is not None else len(a) # hi defaults to len(a)   
        tmp = bisect_left(a, x[i], lo, hi)          # find insertion position
        idxTmp = (tmp if tmp != hi else -1) # don't walk off the end
        idx[i] = idxTmp
    return idx

    
class SpatialView(ManualClusteringView):

    # Some class variables
    n_rate_map_bins = 50
    n_hd_bins = 180
    n_smooth_pos = 2
    n_smooth_hd = 2
    n_px_per_meter = 300
    speed_threshold = 0.025
    tracking_filename = 'tracker_openField.mat'

    default_shortcuts = {
        'go_left': 'alt+left',
        'go_right': 'alt+right',
    }

    def __init__(self,
                 spike_times=None,
                 spike_clusters=None,
                 sample_rate=None,
                 tracking_data=None,
                 **kwargs):

        assert sample_rate > 0

        # Initialize the view.
        super(SpatialView, self).__init__(
            layout='grid',
            shape=(2, 2),
            **kwargs)

        # Spike clusters.
        assert spike_clusters.shape == (self.n_spikes,)
        self.spike_clusters = spike_clusters
        self.sample_rate = float(sample_rate)
        self.spike_times = np.asarray(spike_times)
        self.n_spikes, = self.spike_times.shape
        self.tracking_data = tracking_data

        # extra initialization for spatial functionality
        self.calculate_occupancy_histograms()

    def on_select(self, cluster_ids=None):
        super(SpatialView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return
        
        with self.building():
            for c in range(len(cluster_ids)):
                self.make_plots(c)

    def make_plots(self, clu_idx):
    
            if clu_idx is not None:
                color = tuple(_colormap(clu_idx)) + (.5,)
            else:
                return
            assert len(color) == 4

            cluster_id = cluster_ids[clu_idx]
            # Get indices of all spikes for the current cluster
            idx = np.in1d(self.spike_clusters, cluster_id)

            # Get the spike times, relative to tracking data
            spike_times = self.spike_times[idx] + self.t[0]
            idx = binary_search(self.tracking_data[:, 0], spike_times)
            valid_spikes = idx > 0
            inds_spike_tracking = idx[valid_spikes]
            hd_tuning_curve = self.hd_tuning_curve(inds_spike_tracking)

            pos = self.tracking_data
            x = pos[:, 1]
            y = pos[:, 2]
            
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
            if clu_idx == 0:
                # Both normal and hd-coded
                for i in [0, 1]:
                    self[i, 0].uplot(
                        x=x.reshape((1,-1)),
                        y=y.reshape((1, -1)),
                        color=(1, 1, 1, 0.2),
                        data_bounds=data_bounds
                    )
                    
            # Spike locations
            self[0, 0].scatter(
                x=x[inds_spike_tracking],
                y=y[inds_spike_tracking],
                color=color,
                size=2,
                data_bounds=data_bounds
            )

            # Spike locations (HD-color-coded)
            spike_colors = array_to_rgb(self.spikeHd)
            self[1, 0].scatter(
                x=x[inds_spike_tracking],
                y=y[inds_spike_tracking],
                color=spike_colors,
                size=2,
                data_bounds=data_bounds
                ) 

            # HD plot
            rho = 1.1
            (pol_x, pol_y) = pol2cart(rho, np.linspace(0, 2*math.pi, 1000))
            min_x = np.min(pol_x)
            min_y = np.min(pol_y)
            max_x = np.max(pol_x)
            max_y = np.max(pol_y)
            data_bounds = (min_x, min_y, max_x, max_y)

            # Plot axes first time only
            if clu_idx == 0:
                self[0, 1].uplot(
                        x=pol_x,
                        y=pol_y,
                        color=(1, 1, 1, 0.5),
                        data_bounds = data_bounds
                    )
                    
                self[0, 1].uplot(
                    x=np.array([min_x, max_x]),
                    y=np.array([0, 0]) + (min_y + max_y)/2,
                    color=(1, 1, 1, 0.3),
                    data_bounds = data_bounds
                )
                
                self[0, 1].uplot(
                    y=np.array([min_y, max_y]),
                    x=np.array([0, 0]) + (min_x + max_x)/2,
                    color=(1, 1, 1, 0.3),
                    data_bounds = data_bounds
                )
            
            bin_size_hd = self.bins['hd'][1] - self.bins['hd'][0]
            bin_centres_hd = self.bins['hd'][:-1] + bin_size_hd/2
            (x, y) = pol2cart(hd_tuning_curve, bin_centres_hd)

            # HD tuning curve
            self[0, 1].plot(
                    x=x,
                    y=y,
                    color=color,
                    data_bounds = data_bounds
                )
                
    def attach(self, gui):
        """Attach the view to the GUI."""
        super(SpatialView, self).attach(gui)

    @property
    def state(self):
        return Bunch(n_rate_map_bins=self.n_rate_map_bins,
                     n_hd_bins=self.n_hd_bins,
                     )
        
    def hd_tuning_curve(self, spike_tracking_inds):

        # Get the HD for every spike
        spike_hd = self.hd[spike_tracking_inds]

        # Make the histogram
        tmp = np.histogram(a=spike_hdk, bins=(self.bins['hd']))
        histSpike = tmp[0]

        # Normalize by the occupancy hist to get tuning curve
        crv = histSpike / self.hdOccupancyHist
        badBins = np.isinf(crv) | np.isnan(crv)
        crv[badBins] = 0

        # Gaussian smooth the tuning curve
        crv = sim.filters.gaussian_filter(
                crv,
                sigma=self.n_smooth_hd,
                mode='wrap'
            )
        crv /= np.max(crv)
        return crv
     
    def calculate_occupancy_histograms(self):
        pos = self.tracking_data
        t = pos[:, 0]
        x = pos[:, 1]
        y = pos[:, 2]
        
        # Calculate speed
        dx = np.diff(x)
        dy = np.diff(y)
        dt = dp.diff(t)
        self.speed = np.hstack((0, np.hypot(dy, dx) / dt))
        self.validSpeed = self.speed > self.speed_threshold
        
        # Make the bin vectors for x and y
        self.bins = dict()
        vars = {'x': x, y: 'y'}
        for key, val in vars.items():
            min = np.percentile(val, 0.1)
            max = np.percentile(val, 99.9)
            mid = (max + min) / 2
            range = max - min
            self.bins[key] = np.linspace(mid-range*0.6, mid+range*0.6, self.n_rate_map_bins)
           
        # Make hd bin edge
        self.bins['hd'] = np.linspace(0, 2*math.pi, self.n_hd_bins)
           
        # Calculate the spatial occupancy histogram
        idx = self.validSpeed
        tmp = np.histogram2d(self.x[idx], self.y[idx], bins=(self.bins['x'],self.bins['y']) )
        self.occupancyHist = tmp[0]
        
        # Calculate the HD occupancy histogram
        tmp = np.histogram(self.hd, bins=(self.bins['hd']))
        self.hdOccupancyHist = tmp[0]
        
