import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista

plt.rc('text', usetex=True)
plt.rc('font', size=16)  # use 13 for squared double columns figures
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rc('figure', max_open_warning=0)
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
# mpl.rcParams['font.family'] = 'cm'' #for \text command
plt.style.use('ggplot')
pyvista.set_plot_theme('dark')