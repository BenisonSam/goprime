"""Simple example on how to log scalars and images to tensorboard without tensor ops.
"""
from enum import Enum
from io import StringIO
from multiprocessing import Lock

import numpy as np
import tensorflow as tf

_sync_log_lock = Lock()


class Log(Enum):
    Scalar = 0
    Images = 1
    Histogram = 2


class TBLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self.__sessions = {}

    def log(self, log, session_id, tag, value, **args):
        _sync_log_lock.acquire()
        try:
            if session_id in self.__sessions:
                self.__sessions[session_id] += 1
            else:
                self.__sessions[session_id] = 0

            step = self.__sessions[session_id]
            if log == Log.Scalar:
                self.log_scalar(tag=tag, value=value, step=step)
            elif log == Log.Images:
                self.log_images(tag=tag, images=value, step=step)
            elif log == Log.Histogram:
                self.log_histogram(tag=tag, values=value, step=step, bins=1000 if 'bins' not in args else args['bins'])
        except:
            import traceback
            traceback.print_exc()
        finally:
            _sync_log_lock.release()

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""
        import matplotlib.pyplot as plt

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step).eval()

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
