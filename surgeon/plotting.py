import matplotlib
from matplotlib import pyplot as plt
from . import alluvial

font = {'family': 'Arial',
        # 'weight' : 'bold',
        'size': 14}

matplotlib.rc('font', **font)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('xtick', labelsize=14)


def sankey_diagram(data, save_path="sankey.pdf"):
    plt.close('all')
    cmap = matplotlib.cm.get_cmap('jet')
    ax = alluvial.plot(
        data.tolist(), color_side=1, figsize=(21, 15),
        disp_width=True, wdisp_sep=' '*2, cmap=cmap, v_gap_frac=0.03, h_gap_frac=0.03)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


