import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from .base import tile_image, maryland_cmap



def rotate_right(image):
    image = np.flip(image.transpose((1,0,2)),axis=0)
    return image



def create_fmri_timesteps_plot(
    image,
    template,
    slice_idx=None,
    num_rows=4,
    num_cols=4,
    num_tile_rows=3,
    num_tile_cols=4,
    cmap="inferno",
    colorbar=True,
    zscore = True,
    rotation = 1
):

    if zscore:
        image = st.zscore(image, axis=1)

    subplots_fig, subplots_axes = plt.subplots(
        num_rows, num_cols, figsize=(4 * num_rows, 4 * num_cols)
    )

    plt.tight_layout()

    for i in range(num_rows):
        for j in range(num_cols):
            time_step = num_cols*i + j

            subplot_ax = subplots_axes[i][j]

            if time_step >= image.shape[-1]:
                subplot_ax.set_visible(False)

            else:

                create_tile_plot(
                    image[..., time_step],
                    template,
                    ax=subplot_ax,
                    title=f"t = {time_step}",
                    slice_idx=slice_idx,
                    num_rows=num_tile_rows,
                    num_cols=num_tile_cols,
                    cmap=cmap,
                    colorbar=colorbar
                )
    return subplots_fig, subplots_axes

def create_fmri_factor_plot(
    image,
    template,
    num_tile_rows=3,
    num_tile_cols=4,
    cmap="maryland",
    colorbar=True,
    zscore = True,
    rotation = 1,
    figsize = (4, 4)
):

    if zscore:
        image = st.zscore(image, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    plt.tight_layout()
    create_tile_plot(
        image,
        template,
        ax=ax,
        num_rows=num_tile_rows,
        num_cols=num_tile_cols,
        cmap=cmap,
        colorbar=colorbar
    )

    return fig, ax




def create_tile_plot(
    image,
    template,
    ax,
    title=None,
    slice_idx=None,
    num_rows=3,
    num_cols=4,
    cmap="maryland",
    colorbar=True,
    rotation = 1
):
    if slice_idx is None:
        num_tiles = num_rows * num_cols
        num_slices = image.shape[-1]
        if num_tiles >= num_slices:
            slice_idx = range(num_slices)
        elif num_tiles < num_slices:
            slice_idx = np.linspace(0, num_slices - 1, num_rows * num_cols).astype(int)

    img = image[..., slice_idx]
    for i in range(rotation):
        img = rotate_right(img)
        template = rotate_right(template)

    tiled_fmri = tile_image(img, num_rows=num_rows, num_cols=num_cols)
    tiled_mask = tile_image(
        img.mask, num_rows=num_rows, num_cols=num_cols
    )

    tiled_fmri = np.ma.array(tiled_fmri, mask=tiled_mask)
    tiled_template = tile_image(
        template[..., slice_idx], num_rows=num_rows, num_cols=num_cols
    )
    plot_tiled_fmri(
        tiled_fmri=tiled_fmri,
        ax=ax,
        tiled_template=tiled_template,
        title=title,
        cmap=cmap,
        colorbar=colorbar,
    )


def plot_tiled_fmri(
    tiled_fmri,
    ax,
    tiled_template=None,
    title=None,
    cmap="maryland",
    colorbar=True,
    vmin=None,
    vmax=None
):

    if cmap == "maryland":
        cmap = maryland_cmap

    if tiled_template is not None:
        template_imshow = ax.imshow(tiled_template, cmap="gray")

    if vmax is None:
        vmax = abs(tiled_fmri.max())
    if vmin is None:
        vmin = -vmax

    voxels_imshow = ax.imshow(tiled_fmri, alpha=1, cmap=cmap, vmin=vmin, vmax=vmax)

    if colorbar:
        ax.figure.colorbar(voxels_imshow, shrink=0.8, ax=ax)

    if title is not None:
        ax.set_title(title)

    return voxels_imshow, template_imshow
