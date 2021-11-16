from robust_power_estimation import *

data = io.loadmat("Human_Demo_Data.mat")

standard100 = {}
standard75 = {}
standard50 = {}
robust100 = {}
robust75 = {}
robust50 = {}

"""
    data: input data, shape: (nPnts, nTrials)
    type: estimation type, value='robust'/'standard'
"""
standard100["S"], standard100["f"], standard100["Serr"] = computePowerEstimation(np.squeeze(data["clean100ch16"]), 'standard')
standard75["S"], standard75["f"], standard75["Serr"] = computePowerEstimation(np.squeeze(data["clean75ch16"]), 'standard')
standard50["S"], standard50["f"], standard50["Serr"] = computePowerEstimation(np.squeeze(data["clean50ch16"]), 'standard')


robust100["S"], robust100["f"], robust100["Serr"] = computePowerEstimation(np.squeeze(data["clean100ch16"]), 'robust',err0=3)
robust75["S"], robust75["f"], robust75["Serr"] = computePowerEstimation(np.squeeze(data["clean75ch16"]), 'robust',err0=3)
robust50["S"], robust50["f"], robust50["Serr"] = computePowerEstimation(np.squeeze(data["clean50ch16"]), 'robust',err0=3)

fig, axes = plt.subplots(1, 2, figsize=(15,4))
def displayFunc(ax, data, c,l=''):
    f = data["f"]
    S = 10 * np.log10(data["S"].real)
    Serr1 = 10 * np.log10(data["Serr"][0,:].real)
    Serr2 = 10 * np.log10(data["Serr"][1,:].real)
    
    ax.plot(f, S , color=c, label=l)
    ax.fill_between(f, Serr1, Serr2, color=c, alpha=0.3)
    plt.legend()
    
displayFunc(axes[0], standard100, tuple(np.array([0, 239, 255])/255.0), '100% 去伪影')
displayFunc(axes[0], standard75, tuple(np.array([0, 96, 255])/255.0), '75% 去伪影')
displayFunc(axes[0], standard50, tuple(np.array([0, 0, 143])/255.0), '50% 去伪影')
axes[0].set_xlim(0, 40)
axes[0].set_ylim(-20, 30)
axes[0].set_yticks([-20, -10, 0, 10, 20, 30])
axes[0].set_title('T6-Oz')
axes[0].set_xlabel('频率 (Hz)')
axes[0].set_ylabel('10log10 功率')
axes[0].grid()
axes[0].legend()

displayFunc(axes[1], robust100, tuple(np.array([255, 100, 0])/255.0), '100% 去伪影')
displayFunc(axes[1], robust75, tuple(np.array([239, 30, 30])/255.0), '75% 去伪影')
displayFunc(axes[1], robust50, tuple(np.array([128, 0, 60])/255.0), '50% 去伪影')
axes[1].set_xlim(0, 40)
axes[1].set_ylim(-20, 30)
axes[1].set_yticks([-20, -10, 0, 10, 20, 30])
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
axes[1].set_xlabel('频率 (Hz)')
axes[1].set_ylabel('10log10 功率')
axes[1].set_title('T6-Oz')
axes[1].grid()
axes[1].legend()