import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import numpy as np
import os

Z_dim = [64,128,256,512,1024]
FID = [0,71.3,78.1806,81.7122,85.3522]
IS = [0,5.12981,4.91918,4.54314,4.19091]
nll = [0,13632,12106.1,10195.4,7528.38]

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

def makeFigure(name='Z dim',path='./'):
    fig = plt.figure()
    fig.suptitle(name)
    ax = HostAxes(fig, [0.08, 0.08, 0.70, 0.8])
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.plot(Z_dim,FID,color=cname[0],label='FID')
    ax.set_ylabel('FID')
    ax.set_xlabel('Z dimension')
    ax.legend()
    fig.add_axes(ax)

    def paraAxis(x, y, name,cnt):
        ax_para = ParasiteAxes(ax, sharex=ax)
        ax.parasites.append(ax_para)
        ax_para.axis['right'].set_visible(True)
        ax_paraD = ax_para.get_grid_helper().new_fixed_axis
        ax_para.set_ylabel(name)
        ax_para.axis['right'] = ax_paraD(loc='right',  offset=(40*cnt, 0), axes=ax_para)
        ax_para.plot(x, y, label=name, color=cname[cnt+1])
        plt.legend(loc=0)

    paraAxis(Z_dim, IS, "IS",0)
    paraAxis(Z_dim, nll, "nll",1)
    plt.savefig(os.path.join(path,'z_dim.png'),dpi=100)

makeFigure()