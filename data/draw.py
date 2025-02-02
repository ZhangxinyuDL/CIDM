
import numpy as np
# from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw(data, path):
    cdict = ['whitesmoke', 'dodgerblue', 'limegreen', 'green', 'darkgreen',
             'yellow', 'goldenrod', 'orange', 'red', 'darkred']
    clevs = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    my_map = colors.ListedColormap(cdict)
    norm = colors.BoundaryNorm(clevs, len(clevs) - 1)
    fig = plt.figure()
    ax = plt.gca()
    # 加载地图
    # map_B = Basemap(projection='cyl', llcrnrlon=000, llcrnrlat=000, urcrnrlon=000, urcrnrlat=000,
    #                 resolution='l')
    # map_B.readshapefile(r"", 'Province, drawbounds=True,default_encoding='iso-0000-00')
    # 画经纬度网格线
    # parallels = np.arange(00, 00, 0.5)
    # map_B.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
    # meridians = np.arange(00., 00., 0.5)
    # map_B.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)
    im = ax.imshow(data, cmap=my_map, norm=norm, origin='lower')
  
    drivider = make_axes_locatable(ax)
    cax = drivider.append_axes("right", size="4%", pad=0.2)
    cbar = plt.colorbar(im,cax=cax)
    #cbar = plt.colorbar(im, fraction=0.04,pad=0.04)#shrink=0.8)
    cbar.set_label('Radar reflectivity (dBZ)', size=12)
    plt.savefig(path, dpi=100)
    plt.cla()
    plt.close()
