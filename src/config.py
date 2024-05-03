import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.rc('text', usetex=True)
plt.rc('font', size=16)  # use 13 for squared double columns figures
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rc('figure', max_open_warning=0)
mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 16
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
# mpl.rcParams['font.family'] = 'cm' #for \text command