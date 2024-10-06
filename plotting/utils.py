import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

def generate_color_palette(series, cmap = plt.cm.gist_rainbow, min_v=0., max_v=1.):

    unique_values = series.sort_values().unique()
    n_colors = len(unique_values)
    colors = plt.cm.get_cmap(cmap)(np.linspace(min_v, max_v, n_colors))
    hsv_colors = matplotlib.colors.rgb_to_hsv(colors[:, :3])
    rgb_colors = matplotlib.colors.hsv_to_rgb(hsv_colors)
    hex_colors = np.apply_along_axis(matplotlib.colors.to_hex, 1, rgb_colors)
    return dict(zip(unique_values, hex_colors))