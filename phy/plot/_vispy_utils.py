# -*- coding: utf-8 -*-

"""Plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np

from vispy import app, gloo, config
from vispy.visuals import Visual

from ..utils.array import _unique, _as_array
from ..utils.logging import debug
from ._panzoom import PanZoom


#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------

def _load_shader(filename):
    """Load a shader file."""
    path = op.join(op.dirname(op.realpath(__file__)), 'glsl', filename)
    with open(path, 'r') as f:
        return f.read()


def _tesselate_histogram(hist):
    assert hist.ndim == 1
    nsamples = len(hist)
    dx = 2. / nsamples

    x0 = -1 + dx * np.arange(nsamples)

    x = np.zeros(5 * nsamples + 1)
    y = -1.0 * np.ones(5 * nsamples + 1)

    x[0:-1:5] = x0
    x[1::5] = x0
    x[2::5] = x0 + dx
    x[3::5] = x0
    x[4::5] = x0 + dx
    x[-1] = 1

    y[1::5] = y[2::5] = -1 + 2. * hist

    return np.c_[x, y]


def _enable_depth_mask():
    gloo.set_state(clear_color='black',
                   depth_test=True,
                   depth_range=(0., 1.),
                   # depth_mask='true',
                   depth_func='lequal',
                   blend=True,
                   blend_func=('src_alpha', 'one_minus_src_alpha'))
    gloo.set_clear_depth(1.0)


#------------------------------------------------------------------------------
# Base spike visual
#------------------------------------------------------------------------------

class _BakeVisual(Visual):
    _shader_name = ''
    _gl_draw_mode = ''

    def __init__(self, **kwargs):
        super(_BakeVisual, self).__init__(**kwargs)
        self._to_bake = []
        self._empty = True

        curdir = op.dirname(op.realpath(__file__))
        config['include_path'] = [op.join(curdir, 'glsl')]

        vertex = _load_shader(self._shader_name + '.vert')
        fragment = _load_shader(self._shader_name + '.frag')
        self.program = gloo.Program(vertex, fragment)

    @property
    def empty(self):
        """Specify whether the visual is currently empty or not."""
        return self._empty

    @empty.setter
    def empty(self, value):
        """Specify whether the visual is currently empty or not."""
        self._empty = value

    def set_to_bake(self, *bakes):
        """Mark data items to be prepared for GPU."""
        for bake in bakes:
            if bake not in self._to_bake:
                self._to_bake.append(bake)

    def _bake(self):
        """Prepare and upload the data on the GPU.

        Return whether something has been baked or not.

        """
        if self._empty:
            return
        n_bake = len(self._to_bake)
        # Bake what needs to be baked.
        # WARNING: the bake functions are called in alphabetical order.
        # Tweak the names if there are dependencies between the functions.
        for bake in sorted(self._to_bake):
            # Name of the private baking method.
            name = '_bake_{0:s}'.format(bake)
            if hasattr(self, name):
                getattr(self, name)()
        self._to_bake = []
        return n_bake > 0

    def draw(self):
        """Draw the waveforms."""
        # Bake what needs to be baked at this point.
        self._bake()
        if not self._empty:
            self.program.draw(self._gl_draw_mode)


class BaseSpikeVisual(_BakeVisual):
    """Base class for a VisPy visual showing spike data.

    There is a notion of displayed spikes and displayed clusters.

    """
    def __init__(self, **kwargs):
        super(BaseSpikeVisual, self).__init__(**kwargs)
        self.n_spikes = None
        self._spike_clusters = None
        self._spike_ids = None
        self._cluster_ids = None
        self._cluster_order = None
        self._update_clusters_automatically = True

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    # Data properties
    # -------------------------------------------------------------------------

    def _set_or_assert_n_spikes(self, arr):
        """If n_spikes is None, set it using the array's shape. Otherwise,
        check that the array has n_spikes rows."""
        # TODO: improve this
        if self.n_spikes is None:
            self.n_spikes = arr.shape[0]
        assert arr.shape[0] == self.n_spikes

    def _update_clusters(self):
        self._cluster_ids = _unique(self._spike_clusters)

    @property
    def spike_clusters(self):
        """The clusters assigned to the displayed spikes."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        """Set all spike clusters."""
        value = _as_array(value)
        self._spike_clusters = value
        if self._update_clusters_automatically:
            self._update_clusters()
        self.set_to_bake('spikes_clusters')

    @property
    def cluster_order(self):
        """List of selected clusters in display order."""
        if self._cluster_order is None:
            return self._cluster_ids
        else:
            return self._cluster_order

    @cluster_order.setter
    def cluster_order(self, value):
        value = _as_array(value)
        assert sorted(value.tolist()) == sorted(self._cluster_ids)
        self._cluster_order = value

    @property
    def masks(self):
        """Masks of the displayed spikes."""
        return self._masks

    @masks.setter
    def masks(self, value):
        value = _as_array(value)
        self._set_or_assert_n_spikes(value)
        # TODO: support sparse structures
        assert value.ndim == 2
        assert value.shape == (self.n_spikes, self.n_channels)
        self._masks = value
        self.set_to_bake('spikes')

    @property
    def spike_ids(self):
        """Spike ids to display."""
        if self._spike_ids is None:
            self._spike_ids = np.arange(self.n_spikes).astype(np.uint64)
        return self._spike_ids

    @spike_ids.setter
    def spike_ids(self, value):
        value = _as_array(value)
        self._set_or_assert_n_spikes(value)
        self._spike_ids = value
        self.set_to_bake('spikes')

    @property
    def cluster_ids(self):
        """Cluster ids of the displayed spikes."""
        return self._cluster_ids

    @cluster_ids.setter
    def cluster_ids(self, value):
        """Clusters of the displayed spikes."""
        self._cluster_ids = _as_array(value)

    @property
    def n_clusters(self):
        """Number of displayed clusters."""
        if self._cluster_ids is None:
            return None
        else:
            return len(self._cluster_ids)

    @property
    def cluster_colors(self):
        """Colors of the displayed clusters."""
        return self._cluster_colors

    @cluster_colors.setter
    def cluster_colors(self, value):
        self._cluster_colors = _as_array(value)
        assert len(self._cluster_colors) >= self.n_clusters
        self.set_to_bake('cluster_color')

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_cluster_color(self):
        if self.n_clusters == 0:
            return
        u_cluster_color = self.cluster_colors.reshape((1, self.n_clusters, -1))
        u_cluster_color = (u_cluster_color * 255).astype(np.uint8)
        self.program['u_cluster_color'] = gloo.Texture2D(u_cluster_color)
        debug("bake cluster color", u_cluster_color.shape)


#------------------------------------------------------------------------------
# Axes and boxes visual
#------------------------------------------------------------------------------

class BoxVisual(_BakeVisual):
    """Box frames in a square grid of subplots."""
    _shader_name = 'box'
    _gl_draw_mode = 'lines'

    def __init__(self, **kwargs):
        super(BoxVisual, self).__init__(**kwargs)
        self._n_rows = None

    @property
    def n_rows(self):
        """Number of rows in the grid."""
        return self._n_rows

    @n_rows.setter
    def n_rows(self, value):
        assert value >= 0
        self._n_rows = value
        self._empty = not(self._n_rows > 0)
        self.set_to_bake('n_rows')

    @property
    def n_boxes(self):
        """Number of boxes in the grid."""
        return self._n_rows * self._n_rows

    def _bake_n_rows(self):
        if not self._n_rows:
            return
        arr = np.array([[-1, -1],
                        [-1, +1],
                        [-1, +1],
                        [+1, +1],
                        [+1, +1],
                        [+1, -1],
                        [+1, -1],
                        [-1, -1]]) * .975
        arr = np.tile(arr, (self.n_boxes, 1))
        position = np.empty((8 * self.n_boxes, 3), dtype=np.float32)
        position[:, :2] = arr
        position[:, 2] = np.repeat(np.arange(self.n_boxes), 8)
        self.program['a_position'] = position
        self.program['n_rows'] = self._n_rows
        debug("bake boxes", position.shape)


class AxisVisual(BoxVisual):
    """Subplot axes in a subplot grid."""
    _shader_name = 'ax'

    def __init__(self, **kwargs):
        super(AxisVisual, self).__init__(**kwargs)
        self._positions = (0., 0.)

    def _bake_n_rows(self):
        self.program['n_rows'] = self._n_rows

    @property
    def positions(self):
        """A pair of (x, y) values for the two axes."""
        return self._positions

    @positions.setter
    def positions(self, value):
        assert len(value) == 2
        self._positions = value
        self.set_to_bake('positions')

    def _bake_positions(self):
        if not self._n_rows:
            return
        position = np.empty((4 * self.n_boxes, 4), dtype=np.float32)
        x, y = self._positions
        c = 1.
        arr = np.array([[x, -c],
                        [x, +c],
                        [-c, y],
                        [+c, y]], dtype=np.float32)
        # Positions.
        position[:, :2] = np.tile(arr, (self.n_boxes, 1))
        # Index.
        position[:, 2] = np.repeat(np.arange(self.n_boxes), 4)
        # Axes.
        position[:, 3] = np.tile([0, 0, 1, 1], self.n_boxes)
        self.program['a_position'] = position
        debug("bake ax", position.shape)


#------------------------------------------------------------------------------
# Base spike canvas
#------------------------------------------------------------------------------

class BaseSpikeCanvas(app.Canvas):
    """Base class for a VisPy canvas with spike data.

    Display a main `BaseSpikeVisual` with pan zoom.

    """

    _visual_class = None
    _pz = None

    def __init__(self, **kwargs):
        super(BaseSpikeCanvas, self).__init__(keys='interactive', **kwargs)
        self.visual = self._visual_class()
        self._create_pan_zoom()

    def _create_pan_zoom(self):
        self._pz = PanZoom()
        self._pz.add(self.visual.program)
        self._pz.attach(self)

    def on_draw(self, event):
        """Draw the main visual."""
        self.context.clear()
        self.visual.draw()

    def on_resize(self, event):
        """Resize the OpenGL context."""
        self.context.set_viewport(0, 0, event.size[0], event.size[1])