'''
makeing figure for FID,IS,nll ~ z dim

use like:
    makeFigure(name,path,dpi)
    # figure while save at path/z_dim.png

adding new data:
    add directly into lists of Z_dim,FID,IS,nll below

adding new axis & curve:
    1) add a new list with same length with z dim
    2) plot it using paraAxis(), like line 68,69

'''
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import numpy as np
import os
import math

Z_dim = [6,7,8,9,10]
FID = [78.6269,71.3,78.1806,81.7122,85.3522]
IS = [4.95797,5.12981,4.91918,4.54314,4.19091]
nll = [15063.1,13632,12106.1,10195.4,7528.38]
for i in range(len(nll)):
    nll[i]/=(3072*math.log(2))

cname = [ 
    '#000000', '#0000FF', 
    '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', 
    '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C',
    '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', 
    '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', 
    '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', 
    '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', 
    '#696969', '#1E90FF', '#B22222', '#FFFAF0', '#228B22', 
    '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700', '#DAA520', 
    '#808080', '#008000', '#ADFF2F', '#F0FFF0', '#FF69B4', 
    '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', 
    '#FFF0F5', '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', 
    '#E0FFFF', '#FAFAD2', '#90EE90', '#D3D3D3', '#FFB6C1', 
    '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE', 
    '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#9370DB',
]

def makeFigure(name='Z dim',path='./',dpi=100):
    fig = plt.figure()
    # fig.suptitle(name)
    ax = HostAxes(fig, [0.08, 0.08, 0.70, 0.8])
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.set_xticks(Z_dim)
    ax.plot(Z_dim,FID,color=cname[0],label='FID')
    ax.set_ylabel('FID')
    ax.set_xlabel('log_2(Z dimension)')
    ax.legend()
    fig.add_axes(ax)

    def paraAxis(x, y, name,cnt):
        ax_para = ParasiteAxes(ax, sharex=ax)
        ax.parasites.append(ax_para)
        ax_para.axis['right'].set_visible(True)
        ax_paraD = ax_para.get_grid_helper().new_fixed_axis
        ax_para.set_ylabel(name)
        ax_para.yaxis.set_label_coords(-1,-0.5)
        ax_para.axis['right'] = ax_paraD(loc='right',  offset=(40*cnt, 0), axes=ax_para)
        ax_para.plot(x, y, label=name, color=cname[cnt+1])
        plt.legend(loc=8, ncol=3)

    paraAxis(Z_dim, IS, "IS",0)
    paraAxis(Z_dim, nll, "nll",1)
    plt.savefig(os.path.join(path,'z_dim.png'),dpi=dpi)

# makeFigure()

def tw():
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)


    fig, host = plt.subplots(figsize=(5,4))
    fig.subplots_adjust(left=0.1,right=0.85)

    par1 = host.twinx()
    par2 = host.twinx()
    
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.1))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)
    host
    p1, = host.plot(Z_dim, FID, "b-", label="FID")
    p2, = par1.plot(Z_dim, IS, "r-", label="IS")
    p3, = par2.plot(Z_dim, nll, "g-", label="nll")

    # host.set_xlim(0, 2)
    # host.set_ylim(0, 2)
    # par1.set_ylim(0, 4)
    # par2.set_ylim(1, 65)
    host.set_xlabel("log(Z dim)")
    host.xaxis.set_label_coords(0.95,-0.1)
    host.set_ylabel("FID")
    par1.set_ylabel("IS")
    par1.yaxis.set_label_coords(0.95,0.5)
    par2.set_ylabel("bpd")
    par2.yaxis.set_label_coords(1.15,0.49)

    # host.yaxis.label.set_color(p1.get_color())
    # par1.yaxis.label.set_color(p2.get_color())
    # par2.yaxis.label.set_color(p3.get_color())
    xtkw = dict(size=4, width=1)
    tkw = dict(size=4, width=0.5)
    host.tick_params(axis='y', **tkw)
    host.set_xticks(Z_dim)
    par1.tick_params(axis='y', **tkw)
    par2.tick_params(axis='y', **tkw)
    host.tick_params(axis='x', **xtkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines],loc='upper center',bbox_to_anchor=(0.4,-0.05), ncol=3)

    plt.savefig(os.path.join('.','z_dim.png'),dpi=100)

tw()