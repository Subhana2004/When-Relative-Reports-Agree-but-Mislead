"""Self-contained journal-style figures from numerical outputs; no fabricated values."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':240})
colors=['#334155','#24748e','#9b634d','#2b7556','#b07a29']

# Plot 1: Model-free decoder Brier scores across noise settings.
dec=json.loads((ROOT/'experiments/results/decoder_extension.json').read_text())['regimes']
methods=['tree','unweighted_map','weighted_map','weighted_soft','robust_soft']
labels=['Tree completion','Unweighted MAP','Weighted MAP','Bayes soft','Mixture soft']
fig,axes=plt.subplots(1,2,figsize=(9.8,3.35),sharex=True)
for ax,reg in zip(axes,['heavy','correlated']):
    v=[dec[reg][a]['brier'] for a in methods]
    ax.barh(np.arange(5),v,color=colors,height=.63)
    ax.set_yticks(range(5));ax.set_yticklabels(labels,fontsize=9)
    ax.set_xlim(0,.36);ax.set_title('Independent noise' if reg=='heavy' else 'Correlated noise')
    ax.grid(axis='x',alpha=.18);ax.invert_yaxis()
    ax.set_xlabel('Brier score (lower is better)')
fig.tight_layout()
fig.savefig(OUT/'decoder_brier.pdf',bbox_inches='tight');fig.savefig(OUT/'decoder_brier.png',bbox_inches='tight');plt.close(fig)

# Graph gauge invariance conceptual diagram; labels are relation bits, not graph edges
from matplotlib.patches import FancyArrowPatch
fig,axs=plt.subplots(1,2,figsize=(8.4,3.1))
coord=np.array([[0,.72],[-.75,-.55],[0,-.62],[.75,-.55]])
for k,ax in enumerate(axs):
    bits={(0,1):1,(0,2):0,(0,3):1,(1,2):1,(1,3):0,(2,3):1}
    if k==1:bits={e:(v ^ int(0 in e)) for e,v in bits.items()}
    for (a,b),bit in bits.items():
        x,y=coord[a],coord[b]
        ax.plot([x[0],y[0]],[x[1],y[1]],color='#b9c3cc',lw=1.4,zorder=1)
        mid=(x+y)/2
        ax.text(mid[0]+.025,mid[1]+.025,str(bit),fontsize=8,color='#475569',backgroundcolor='white')
    for i,(x,y) in enumerate(coord):
        ax.scatter(x,y,s=480,color=('#186b77' if i==0 else '#64748b'),zorder=3)
        ax.text(x,y,str(i),ha='center',va='center',color='white',weight='bold',fontsize=10,zorder=4)
    ax.set_title('Reported parities' if k==0 else 'Anchor-fan flip')
    ax.set_xlim(-1.0,1.0);ax.set_ylim(-.95,1.02);ax.set_aspect('equal');ax.axis('off')
fig.text(.5,.01,'Triangle XOR syndromes are unchanged, despite the three altered anchor-adjacent reports.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(OUT/'gauge.pdf',bbox_inches='tight');fig.savefig(OUT/'gauge.png',bbox_inches='tight');plt.close(fig)

def model_plots():
    d=json.loads((ROOT/'experiments/results/extension_summary.json').read_text())
    p=d['paired'];g=d['groups']
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.5),sharey=True)
    for ax,reg,lab in zip(axs,['heavy','correlated'],['Independent noise','Correlated noise']):
        pair=p[reg];scores=list(pair['seed_difference_means'].values());vals=np.array(scores)*100
        xs=np.arange(1,len(vals)+1)
        ax.axhline(0,color='#64748b',lw=1,ls='--')
        ax.vlines(xs,0,vals,colors=['#24748e' if t>0 else '#aa6150' if t<0 else '#64748b' for t in vals],lw=2,alpha=.85)
        ax.scatter(xs,vals,s=24,color=['#24748e' if t>0 else '#aa6150' if t<0 else '#64748b' for t in vals])
        ax.set_xticks(xs[::2]);ax.set_xticklabels([str(x) for x in list(pair['seed_difference_means'].keys())[::2]],fontsize=8)
        ax.set_xlabel('Training seed');ax.set_title(lab);ax.grid(axis='y',alpha=.13)
        ax.set_ylim(-105,105)
    axs[0].set_ylabel('Joint minus soft-label CQA (pp)')
    fig.tight_layout();fig.savefig(OUT/'seed_diffs.pdf',bbox_inches='tight');fig.savefig(OUT/'seed_diffs.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.4),sharey=False)
    for ax,reg,lab in zip(axs,['heavy','correlated'],['Independent noise','Correlated noise']):
        arms= d['protocol']['arms'][reg]
        names={'anchor':'Anchor','unweighted_map':'Unweighted MAP','em_soft':'EM soft','weighted_soft':'Bayes soft','joint_independent':'Joint','oracle':'Oracle labels','robust_soft':'Mixture soft','joint_robust':'Mixture joint'}
        means=[100*g[reg][a]['mean'] for a in arms]
        ax.barh(range(len(arms)),means,color=[colors[(i+1)%len(colors)] for i in range(len(arms))]);ax.set_yticks(range(len(arms)));ax.set_yticklabels([names[a] for a in arms],fontsize=9)
        ax.set_xlim(0,100);ax.grid(axis='x',alpha=.14);ax.invert_yaxis();ax.set_title(lab);ax.set_xlabel('Complete-quadruple accuracy (%)')
        for i,v in enumerate(means):ax.text(min(v+1.1,98),i,f'{v:.1f}',va='center',ha='left' if v<92 else 'right',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'neural_baselines.pdf',bbox_inches='tight');fig.savefig(OUT/'neural_baselines.png',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
    if (ROOT/'experiments/results/extension_summary.json').exists():model_plots()
