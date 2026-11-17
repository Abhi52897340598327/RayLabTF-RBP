#!/usr/bin/env python3
"""Screen mouse RNA-seq catalogs and summarize sourced NOVA2 brain events."""
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/mouse'; OUT=ROOT/'graphs/mouse/brain_splicing'
(OUT/'figures').mkdir(parents=True,exist_ok=True); (OUT/'tables').mkdir(parents=True,exist_ok=True)
brain=re.compile(r'\b(brain|cortex|cortical|cerebell|hippocamp|forebrain|midbrain|hindbrain|hypothalam|striat|neocort|amygdala|thalam|neuron|neuronal)\w*',re.I)
perturb=re.compile(r'\b(knock.?out|knock.?down|ko|cko|deletion|deleted|deficien\w*|mutant|crispr|shrna|sirna|overexpress\w*)\b',re.I)
control=re.compile(r'\b(control|ctrl|wild.?type|wt|vehicle|scramble\w*|non.target\w*)\b',re.I)
frames=[]
for kind,folder in [('TF','Mouse_TF_Datasets_csv'),('RBP','Mouse_RBP_Datasets (2)_csv')]:
 for p in sorted((DATA/folder).glob('RNA-Seq*.csv')):
  for df in pd.read_csv(p,dtype=str,chunksize=50000,low_memory=False):
   df=df.fillna(''); matches=df.Tissue.str.contains(brain,na=False) | df.Title.str.contains(brain,na=False)
   df=df[matches].copy(); df['catalog_class']=kind; df['catalog_file']=p.name
   df['brain_match_location']=np.where(df.Tissue.str.contains(brain,na=False),'tissue','title only')
   df['title_perturbation_keyword']=df.Title.str.contains(perturb,na=False)
   df['title_control_keyword']=df.Title.str.contains(control,na=False)
   frames.append(df)
screen=pd.concat(frames,ignore_index=True).drop_duplicates()
screen.to_csv(OUT/'tables/catalog_brain_rnaseq_screen.csv',index=False)
screen.groupby(['catalog_class','brain_match_location']).size().rename('catalog_rows').to_csv(OUT/'tables/catalog_brain_rnaseq_screen_summary.csv')
# Files are already significant results from the authors, not all tested events.
files={'excitatory_cortex':'cortex_excitatory.tsv.gz','inhibitory_cortex':'cortex_inhibitory.tsv.gz','cerebellum_purkinje':'cerebellum.tsv.gz'}
keys=['chr','chr.Start','chr.End','strand','AS.type','AS.name','isoformIDs']
event_sets={}; all_events=[]
for comparison,filename in files.items():
 p=DATA/'studies/GSE103314/source'/filename
 df=pd.read_csv(p,sep='\t',compression='gzip')
 assert (df.FDR<0.05).all() and (df['delta.exon.inclusion.rate(dI)'].abs()>=0.1-1e-8).all()
 df['comparison']=comparison
 df['event_key']=df[keys].astype(str).agg('|'.join,axis=1)
 df[['ensembl_gene','gene_symbol']]=df['Ensemble_gene//Gene_symbol'].str.split('//',n=1,expand=True)
 df.to_csv(OUT/'tables'/f'GSE103314_{comparison}_events.csv',index=False)
 event_sets[comparison]=set(df.event_key); all_events.append(df)
combined=pd.concat(all_events,ignore_index=True)
counts=[]
for comparison,group in combined.groupby('comparison'):
 counts.append({'comparison':comparison,'significant_rows':len(group),'unique_event_keys':group.event_key.nunique(),'unique_gene_labels':group.gene_symbol.nunique()})
pd.DataFrame(counts).to_csv(OUT/'tables/GSE103314_comparison_summary.csv',index=False)
types=combined.groupby(['comparison','AS.type']).size().unstack(fill_value=0)
types.to_csv(OUT/'tables/GSE103314_event_types.csv')
intersection=[]
for a in files:
 for b in files:
  intersection.append({'comparison_a':a,'comparison_b':b,'shared_event_keys':len(event_sets[a]&event_sets[b]),'jaccard':len(event_sets[a]&event_sets[b])/len(event_sets[a]|event_sets[b])})
pd.DataFrame(intersection).to_csv(OUT/'tables/GSE103314_event_overlap.csv',index=False)
shared=combined[combined.event_key.isin(set.union(*(event_sets[a]&event_sets[b] for i,a in enumerate(files) for b in list(files)[i+1:])))].sort_values(['event_key','comparison'])
shared.to_csv(OUT/'tables/GSE103314_shared_events.csv',index=False)
f,ax=plt.subplots(figsize=(9,5)); types.plot.bar(ax=ax,stacked=True,colormap='tab20'); ax.set_ylabel('Published significant event rows'); ax.set_xlabel('Comparison'); ax.set_title('NOVA2 knockout: event types in mouse brain'); ax.tick_params(axis='x',rotation=15); ax.legend(title='Quantas event type',bbox_to_anchor=(1.02,1)); f.tight_layout(); f.savefig(OUT/'figures/GSE103314_event_type_counts.png',dpi=200,bbox_inches='tight'); plt.close(f)
mat=pd.DataFrame(index=files,columns=files,dtype=float)
for x in intersection: mat.loc[x['comparison_a'],x['comparison_b']]=x['jaccard']
f,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=1); ax.set_xticks(range(3),mat.columns,rotation=25,ha='right'); ax.set_yticks(range(3),mat.index)
for i in range(3):
 for j in range(3): ax.text(j,i,f'{mat.iloc[i,j]:.2f}',ha='center',va='center',color='white' if mat.iloc[i,j]>.5 else 'black')
ax.set_title('NOVA2 knockout: Jaccard of reported significant events'); f.colorbar(im,ax=ax,label='Jaccard similarity'); f.tight_layout(); f.savefig(OUT/'figures/GSE103314_significant_event_jaccard.png',dpi=200,bbox_inches='tight'); plt.close(f)
print('Brain RNA-seq catalog candidate rows:',len(screen)); print(pd.DataFrame(counts).to_string(index=False)); print('Pair overlaps:',intersection)
