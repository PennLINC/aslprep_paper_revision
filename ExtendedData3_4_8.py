import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import scipy.stats
from scipy import stats
import scipy
import matplotlib.pylab as plt
import matplotlib
import nilearn.plotting as nip
import nilearn.image as nim
from nilearn.masking import apply_mask
# matplotlib.rcParams['pdf.fonttype'] = 42 #for editable pdf figures
"""
SFigure3
"""
sns.set(style="whitegrid",font="Palatino")


df = pd.read_csv('ExtendedDataFigure3/ExtendedDataFigure3.csv').rename(columns={'newfwhm':'ASLPrep','oldfwhm':'Previous Pipeline'})
print (df.shape)
plt.close()
jp = sns.jointplot(y='ASLPrep',x='Previous Pipeline',data=df,xlim=[3,11],ylim=[3,11],**{"s":10,'alpha':0.75})

df = df.dropna()
print (pearsonr(df.ASLPrep,df['Previous Pipeline']))


print (scipy.stats.ttest_rel(df.ASLPrep,df['Previous Pipeline']))
jp.ax_marg_x.grid(False)
jp.ax_marg_y.grid(False)

slope = 1
intercept = 0
axes = jp.ax_joint
plt.sca(axes)
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')
plt.suptitle("FWHM (mm)") 
plt.savefig('ExtendedDataFigure3/SFigure3.pdf')

"""
SFigure4
"""

def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),xy=(.9, .1), xycoords=ax.transAxes)

cbf = nim.load_img("ExtendedDataFigure4/mean_cbf.nii.gz")
pet = nim.load_img("ExtendedDataFigure4/PET.nii")
mask = nim.load_img("ExtendedDataFigure4/CBF_Mask.nii.gz")
pet,cbf = apply_mask([pet,cbf], mask_img=mask)

try: null_r = np.load('ExtendedDataFigure4/null_values.npy')
except:
    sm = np.load('ExtendedDataFigure4/surrogate_maps.npy')
    null_r = []
    for m in sm:
        null_r.append(stats.pearsonr(m,pet)[0])
    np.save('ExtendedDataFigure4/null_values.npy',null_r)

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
plt.savefig('ExtendedDataFigure4/SFigure4.pdf')


"""
SFigure 8
"""

comptimed = pd.read_csv('ExtendedDataFigure8/ExtendedDataFig8.csv')
comptimed[(comptimed.modality=='anat')&(comptimed['compute time(hrs)']<2)] = np.nan
comptimed[(comptimed.modality=='perf')&(comptimed['compute time(hrs)']<0.16666)] = np.nan

plt.figure(figsize=(8, 5),dpi=600)
bb = sns.stripplot(x="Datasets", y='compute time(hrs)',hue='modality',data=comptimed,jitter=.2,alpha=.35
                  )

bb.set_xlabel('Datasets',fontsize=20)
bb.set_ylabel('Compute Time (hrs)',fontsize=20)
bb.tick_params(labelsize=20)
plt.legend(fontsize='20', title_fontsize='10')
plt.tight_layout()
plt.savefig('ExtendedDataFigure8/ExtendedDataFig8.pdf')
