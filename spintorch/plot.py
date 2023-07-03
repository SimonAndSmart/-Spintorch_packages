import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm
from matplotlib.ticker import MaxNLocator
from .geom import WaveGeometryMs, WaveGeometry
from .solver import MMSolver

import warnings
warnings.filterwarnings("ignore", message=".*No contour levels were found.*")


mpl.use('Agg',) # uncomment for plotting without GUI
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 600


def plot_loss(loss_iter, plotdir):
    fig = plt.figure()
    if len(loss_iter)>=200:    # changes here _sinan
        plt.plot(loss_iter, 'x-')
    else:
        plt.plot(loss_iter, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'_loss.png')
    plt.close(fig)
    
def plot_loss_dict(loss_dict, plotdir, exclude_keys=['min_loss']):   # function added _sinan
    dist = {key: [np.nan if value is None else value for value in values]
            for key, values in loss_dict.items() if key not in exclude_keys}  # convert None into NaN and exclude specific keys
    
    fig = plt.figure()
    for label in dist.keys():
        plt.plot(dist[label], marker = '.' ,label=f'label:{label}')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'_separate_loss.png')
    plt.close(fig)

    
def plot_output(u, label, epoch, plotdir): #changes here _sinan
    fig = plt.figure(dpi=200)
    plt.bar(range(0,u.size()[0]), u.detach().cpu().squeeze(), color='k') # range has been changed from (1,probe_no+1) to (0,probe_no)
    plt.xlabel("output number")
    plt.ylabel("output")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+f'output_epoch{epoch}_label{label}.png') 
    plt.close(fig)
    
def plot_multiple_outputs(probe_reading, labels,epoch_batch, plotdir): #added function _sinan
    num_plots = len(probe_reading)
    ncols = min(4,len(labels)) # max 4 in a column
    nrows = num_plots // ncols if num_plots % ncols == 0 else num_plots // ncols + 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4),sharey=True,dpi=150) # Change figsize as needed
    
    if num_plots == 1:# If there is only one subplot, put it in a list to make it iterable
        axs = [axs]
    else:# Otherwise, flatten axs to iterate over it
        axs = axs.flatten()  # Flattening to allow easy iteration

    # plot each output in a separate subplot
    for sig in range(num_plots):
        u = probe_reading[sig]
        label = int(labels[sig])
        axs[sig].bar(range(0,u.size()[0]), u.detach().cpu().squeeze(), color='k')
        axs[sig].set_xlabel("output number")
        axs[sig].set_ylabel("output")
        axs[sig].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[sig].set_title(f'Label: {label}')

    # Remove unused subplots
    if num_plots % ncols != 0:
        for idx in range(num_plots, ncols*nrows):
            fig.delaxes(axs[idx])

    fig.suptitle(f'{epoch_batch}', fontsize=16)
    fig.tight_layout()
    fig.savefig(plotdir+f'output_{epoch_batch}_all.png')
    plt.close(fig)

def _plot_probes(probes, ax,probe_center): # changes here _Sinan
    markers = []
    for i, probe in enumerate(probes):
        x,y = probe.coordinates()
        if probe_center and [int(x),int(y)] in probe_center:
            marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='r',markersize=4,alpha=0.9)
        else:
            marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='k',markersize=4,alpha=0.8)
        markers.append(marker)
    return markers


def _plot_sources(sources, ax):
    markers = []
    for i, source in enumerate(sources):
        x,y = source.coordinates()
        marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='g',markersize=4,alpha=0.8)
        markers.append(marker)
    return markers


def geometry(model, ax=None, outline=False, outline_pml=True, epoch=0, plotdir='',probe_center_list=None): #changes here _sinan

    geom = model.geom
    probes = model.probes
    sources = model.sources
    A = model.Alpha()[0, 0, ].squeeze()
    alph = A.min().cpu().numpy()
    B = geom.B[1,].detach().cpu().numpy().transpose()

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

    markers = []
    if not outline:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            h1 = ax.imshow(Msat, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, label='Saturation magnetization (A/m)')
        else:
            h1 = ax.imshow(B*1e3, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, label='Magnetic field (mT)')
    else:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            ax.contour(Msat, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)
        else:
            ax.contour(B, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)

    if outline_pml:
        b_boundary = A.cpu().numpy().transpose()
        ax.contour(b_boundary, levels=[alph*1.0001], colors=['k'], linestyles=['dotted'], linewidths=[0.75], alpha=1)

    markers += _plot_probes(probes, ax,probe_center=probe_center_list)
    markers += _plot_sources(sources, ax)
        
    if plotdir:
        fig.savefig(plotdir+'geometry_epoch%d.png' % (epoch))
        plt.close(fig)


def wave_integrated(model, m_history, filename='',probe_center_list=None): # changes here _sinan
    
    m_int = m_history.pow(2).sum(dim=0).numpy().transpose()
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    vmax = m_int.max()
    h = ax.imshow(m_int, cmap=plt.cm.viridis, origin="lower", norm=LogNorm(vmin=vmax*0.01,vmax=vmax))
    plt.colorbar(h)
    geometry(model, ax=ax, outline=True,probe_center_list=probe_center_list)

    if filename:
        fig.savefig(filename)
        plt.close(fig)


def wave_snapshot(model, m_snap, filename='', clabel='m'):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    m_t = m_snap.cpu().numpy().transpose()
    h = axs.imshow(m_t, cmap=plt.cm.RdBu_r, origin="lower", norm=CenteredNorm())
    geometry(model, ax=axs, outline=True)
    plt.colorbar(h, ax=axs, label=clabel, shrink=0.80)
    axs.axis('image')
    if filename:
        fig.savefig(filename)
        plt.close(fig)
        
