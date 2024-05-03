import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import src.config

def plot_history(history, loss_str=None, fname="img/history.pdf"):
    """
    Plots training history loss and error for training and validation datasets.
    """
    fig, ax1 = plt.subplots()

    epochs = np.fromiter(range(len(history['train_loss'])), dtype=int)
    train_loss = np.fromiter(history['train_loss'], dtype=float)
    val_loss = np.fromiter(history['val_loss'], dtype=float)
    val_err = np.fromiter(history['val_error'], dtype=float)

    ax1.plot(epochs, train_loss, color='grey', linewidth=1, label='$L_\mathrm{train}$')
    ax1.plot(epochs, val_loss, color='black', linewidth=1, label='$L_\mathrm{val}$')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(epochs, val_err*100, color='blue', linewidth=1, label='$E_\mathrm{val}$')

    max_train_loss = np.ceil(np.log10(np.max(train_loss)))
    min_train_loss = np.floor(np.log10(np.min(train_loss)))
    max_val_loss = np.ceil(np.log10(np.max(val_loss)))
    min_val_loss = np.floor(np.log10(np.min(val_loss)))

    ax1.set_xlim(np.min(epochs), np.max(epochs) if np.max(epochs) > 0 else 1)
    ax1.set_ylim(10**(np.min([min_train_loss,min_val_loss])), 10**(np.max([max_train_loss,max_val_loss])))
    ax1.set_yscale('log')
    ax1.set_xlabel('$\mathrm{Epoch}$')
    ax1.set_ylabel('$\mathrm{Loss}$' if loss_str is None else
                   '$\mathrm{Loss\,\,('+''.join(loss_str.split('_loss')[:-1]).replace('_','\_').upper()+')}$')
    ax2.set_ylabel('$\% \mathrm{Error}$')
    ax1.grid(axis='both', alpha=0.5)
    # if best_epoch < max_epochs: ax2.vlines(x=best_epoch, ymin=0, ymax=1, linewidth=1, color='r', ls='--')

    ax1.tick_params(bottom=True, top=True, right=False, which='both', direction='in', length=2)
    ax2.tick_params(bottom=False, top=False, right=True, which='both', direction='in', length=2)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    leg = fig.legend(loc='upper right',bbox_to_anchor=(1,1),bbox_transform=ax1.transAxes)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.85)

    if isinstance(fname, str):
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return

def plot_dataset_points(dataset, i=None, fname=None):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(dataset.data["x"], dataset.data["y"])
    if i is not None: ax.plot(*dataset.panels[i][:,:2].T, marker='x', markersize=3, color='b')
    if isinstance(fname, str):
        mpl.use('Agg') # disables interactive backend
        plt.savefig(fname, transparent=True, bbox_inches='tight')
    else:
        plt.show()
    return

def plot_quantity(q, x=None, y=None, label=None,fname=None):
    fig, ax = plt.subplots(figsize=(11,9))
    im = ax.imshow(q, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()), cmap='RdBu')
    aspect = np.abs((x.max()-x.min())/(y.max()-y.min()))
    ax.set_aspect(aspect)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    cbar = plt.colorbar(im)
    cbar.formatter.set_powerlimits((0, 0))
    if label is not None:
        x0, y0 = (x.max()-x.min())*.9+x.min(),(y.max()-y.min())*.9+y.min()
        ax.text(x0, y0,label,fontsize=30, ha='center', va='center', color='white')#, path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=256)
    return fig, ax