# %pylab inline
import nilearn.plotting as nip
import nilearn.image as nim
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import seaborn as sns
import pandas as pd
from nilearn.masking import apply_mask
from scipy import stats
import numpy as np
import scipy
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),xy=(.9, .1), xycoords=ax.transAxes)

sm = np.load('surrogate_maps.npy')
cbf = nim.load_img("mean_cbf.nii.gz")
pet = nim.load_img("PET.nii")
mask = nim.load_img("CBF_Mask.nii.gz")
pet,cbf = apply_mask([pet,cbf], mask_img=mask)



null_r = []
for m in sm:
    null_r.append(stats.pearsonr(m,pet)[0])


plt.close()
sns.set(style="whitegrid",font="Palatino")
f,axes = plt.subplots(1,2)
f.set_size_inches(7.5,3.75)
plt.sca(axes[0])
x,y = pet,cbf
x = scipy.stats.zscore(pet)
y = scipy.stats.zscore(cbf)
# plt.hexbin(x, y, gridsize=25, edgecolor='grey',cmap='inferno',vmax=1000, mincnt=1)
choice = np.random.choice(range(0,x.shape[0]),5000)
sns.scatterplot(x[choice],y[choice],alpha=0.15,ax=axes[0],linewidth=0,**{'edgecolors':'none'})
slope = 1
intercept = 0
x_vals = np.array(axes[0].get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')
sns.despine()
corrfunc(x,y)

plt.xlabel('PET (z-score)')
plt.ylabel('CBF (z-score)')
axes[0].set_ylim(-3,3)
axes[0].set_xlim(-3,3)

axes[0].set_yticks(range(-3,4))
axes[0].set_yticklabels(range(-3,4))
axes[0].set_xticks(range(-3,4))
axes[0].set_xticklabels(range(-3,4))
plt.sca(axes[1])
sns.kdeplot(null_r,clip_on=False,bw_adjust=.5,fill=True, alpha=.5, linewidth=0,color='black',ax=axes[1])
plt.axvline(0.6,0,1,linewidth=2)
plt.xlabel('PET-CBF correlation (r)')
axes[0].set_title('a',{'fontweight':'bold'},'left')
axes[1].set_title('b',{'fontweight':'bold'},'left')
plt.tight_layout()
axes[0].set_ylim(-3,3)
axes[0].set_xlim(-3,3)
plt.savefig('pet_cbf.pdf')