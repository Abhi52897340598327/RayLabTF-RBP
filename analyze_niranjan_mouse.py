#!/usr/bin/env python3
"""Analyze Niranjan fly splicing events and mouse dataset catalogs.

Run from the repository root: MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python analyze_niranjan_mouse.py
"""
from pathlib import Path
from collections import defaultdict
import itertools
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import script as fly
import plot_tf_rbp_stacked_gene_charts as charts

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
GRAPHS = ROOT / 'graphs'


def niranjan():
    base = GRAPHS / 'niranjan'
    fig = base / 'figures'; tables = base / 'tables'
    fig.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    frames = {}
    sets = {}
    for kind in ('TF', 'RBP'):
        path = DATA / 'niranjan' / f'{kind}_master_brain_csv' / 'All events.csv'
        df = pd.read_csv(path).rename(columns={kind: 'dataset_name'})
        df = df.dropna(subset=['dataset_name', 'event_id']).copy()
        df['source_file'] = str(path.relative_to(ROOT))
        frames[kind] = df
        sets[kind] = {str(name): set(g.event_id.astype(str)) for name, g in df.groupby('dataset_name')}
    stats = fly.calculate_tf_rbp_overlap(sets['TF'], sets['RBP']).sort_values(['Jaccard_similarity','Shared_exact_events'], ascending=False)
    stats.to_csv(tables / 'tf_rbp_overlap_statistics.csv', index=False)
    stats.head(15).to_csv(tables / 'top_tf_rbp_overlaps.csv', index=False)
    counts, jac = fly.plot_tf_rbp_heatmap(stats, fig)
    counts.to_csv(tables / 'tf_rbp_shared_event_count_matrix.csv')
    jac.to_csv(tables / 'tf_rbp_jaccard_matrix.csv')
    # Construct the same pair-by-event evidence table without repeatedly scanning
    # all events for every regulator pair.
    lookups = {}
    for kind, df in frames.items():
        lookups[kind] = {(str(reg), str(ev)): fly.summarize_event_rows(group)
                         for (reg, ev), group in df.groupby(['dataset_name','event_id'], sort=False)}
    records = []
    for pair in stats.itertuples(index=False):
        for event in sorted(sets['TF'][pair.TF] & sets['RBP'][pair.Regulator]):
            t = lookups['TF'][(pair.TF,event)]; r = lookups['RBP'][(pair.Regulator,event)]
            row = {'TF':pair.TF,'RBP':pair.Regulator,'event_id':event,
                   'shared_exact_events_in_pair':pair.Shared_exact_events,'jaccard_similarity':pair.Jaccard_similarity,
                   'tf_event_count':pair.TF_events,'rbp_event_count':pair.Regulator_events,
                   'gene_id':t.get('gene_id',''),'gene_name':t.get('gene_name',''),'event_type':t.get('event_type','')}
            for prefix, summary in [('tf',t),('rbp',r)]:
                for key in ('experiments','experiment_count','dPSI_values','p_values','source_files','row_count'):
                    row[f'{prefix}_{key}'] = summary.get(key, '')
            records.append(row)
    pd.DataFrame(records).to_csv(tables / 'tf_rbp_jaccard_important_sites.csv', index=False)
    age_path = DATA / 'fly' / '03_all_age_sex_splicing_events.csv'
    age_frames = fly.load_age_files(age_path.parent)
    candidates = fly.build_candidate_age_table(sets['TF'],sets['RBP'],age_frames)
    candidates.drop(columns=['pattern_order']).to_csv(tables / 'tf_rbp_candidate_age_sex_associations.csv',index=False)
    fly.plot_age_dpsi_heatmap(candidates,fig)
    fly.plot_age_persistence_summary(candidates,fig)
    sex_summary = fly.classify_sex_direction_by_pattern(candidates)
    sex_summary.to_csv(tables / 'tf_rbp_candidate_age_persistence_by_sex.csv',index=False)
    for column, name in [('Male-higher','tf_rbp_candidate_age_persistence_male_higher.png'),('Female-higher','tf_rbp_candidate_age_persistence_female_higher.png')]:
        fly.plot_single_direction_bar(sex_summary,column,name,fig)
    fly.plot_combined_sex_direction(sex_summary,fig)
    strict = charts.load_strict_sex_specific_events(age_path)
    # Existing chart functions take paths in the standardized fly schema.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='niranjan_analysis_') as temp:
        paths = {}
        for kind in ('TF','RBP'):
            paths[kind] = Path(temp) / f'{kind}.csv'; frames[kind].to_csv(paths[kind],index=False)
        for kind in ('TF','RBP'):
            summary = charts.build_regulator_gene_summary(paths[kind],strict,kind)
            summary.to_csv(tables / f'{kind.lower()}_spliced_gene_summary.csv',index=False)
            charts.plot_grouped_count_chart(summary,fig / f'{kind.lower()}_absolute_grouped_spliced_genes.png',x_rotation=60)
            charts.plot_percentage_stacked_chart(summary,fig / f'{kind.lower()}_100_percent_stacked_spliced_genes.png',x_rotation=60)
        summary = charts.build_regulator_class_summary(paths['TF'],paths['RBP'],strict)
        summary.to_csv(tables / 'tf_rbp_class_spliced_gene_summary.csv',index=False)
        charts.plot_grouped_count_chart(summary,fig / 'tf_rbp_class_absolute_grouped_spliced_genes.png',x_rotation=0)
        charts.plot_percentage_stacked_chart(summary,fig / 'tf_rbp_class_100_percent_stacked_spliced_genes.png',x_rotation=0)
    # Sex-set workbooks describe sex-limited experiments; they are separate from
    # the independent male-versus-female age comparisons above.
    sexrows=[]
    for kind in ('TF','RBP'):
        sex = pd.read_csv(DATA / 'niranjan' / f'{kind}_master_sex_specific_csv' / 'All events.csv')
        for (reg, label), group in sex.dropna(subset=[kind,'event_id','Sex set']).groupby([kind,'Sex set']):
            ev=set(group.event_id.astype(str)); brain=sets[kind].get(str(reg),set())
            sexrows.append({'regulator_type':kind,'regulator':reg,'sex_set':label,'unique_events_in_sex_set':len(ev),
                            'unique_events_in_brain_set':len(brain),'exact_events_shared_with_brain_set':len(ev & brain)})
    pd.DataFrame(sexrows).to_csv(tables / 'sex_limited_experiment_event_overlap.csv',index=False)
    (base/'README.md').write_text('''# Niranjan fly splicing analysis\n\nInputs: TF and RBP brain workbooks, `All events` sheets. Exact event IDs are deduplicated within each regulator, pooling studies and experiments. Original and new comparisons may overlap; counts do not imply independent evidence. Source flags and experiment details remain in the source files and site table.\n\nAge/sex plots match exact events against the existing fly 3d/6d/9d male-versus-female dataset and its strict-significance flag, using the original fly code. These age comparisons are independent inputs, not the Niranjan sex-limited experiment sheets. The sex-limited sheets are summarized separately by matching event ID and regulator; “female only” and “male only” describe the experiment sets, not proof of sex-specific splicing. The original brain inputs include mixed/unspecified sex and different ages.\n\nThe gene charts classify genes by whether at least one regulator-associated event matches a strictly significant age/sex event; the remaining genes have no such match, not proven absence of sex specificity. Tables accompany figures in `tables/`.\n''')
    print(f'Niranjan: {len(sets["TF"])} TFs, {len(sets["RBP"])} RBPs, {len(records)} pair-event overlaps, {len(candidates)} age-matched candidate events',flush=True)


def bar(values, title, xlabel, path, color='#4169a1'):
    values = values.sort_values().tail(25)
    fig, ax = plt.subplots(figsize=(10,max(4,0.30*len(values)+1.6)))
    values.plot.barh(ax=ax,color=color)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(''); ax.grid(axis='x',alpha=.2); fig.tight_layout()
    fig.savefig(path,dpi=200,bbox_inches='tight'); plt.close(fig)


def mouse():
    base=GRAPHS/'mouse'; fig=base/'figures'; tables=base/'tables'
    fig.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)
    catalog={}
    for kind, folder in [('TF','Mouse_TF_Datasets_csv'),('RBP','Mouse_RBP_Datasets (2)_csv')]:
        directory=DATA/'mouse'/folder
        manifest=pd.read_csv(directory/'_sheet_manifest.csv')
        paths=[directory / f for f in manifest.csv_file if not re.match(r'^(README|SUMMARY|TECHNIQUE_COUNTS|SOURCE_QC)\.csv$',f,re.I)]
        chunks=[]
        for path in paths:
            for chunk in pd.read_csv(path,usecols=lambda c:c in ['Gene','Accession','Source','Technique','Tissue','Sex'],dtype=str,chunksize=100000,low_memory=False):
                chunk['sheet']=path.stem
                chunks.append(chunk.dropna(subset=['Gene','Accession']))
        df=pd.concat(chunks,ignore_index=True).fillna('')
        for col in ('Gene','Accession','Source','Technique','Tissue','Sex'):
            if col not in df: df[col]=''
            df[col]=df[col].str.strip()
        df=df[(df.Gene!='') & (df.Accession!='')].copy()
        catalog[kind]=df
        # Dataset accession overlap is a catalog match; identifiers of different
        # levels/sources are not assumed equivalent.
        gene_counts=df.groupby('Gene').Accession.nunique().sort_values(ascending=False).rename('unique_accessions').reset_index()
        gene_counts.to_csv(tables/f'{kind.lower()}_accessions_by_gene.csv',index=False)
        tech=df.groupby('Technique').agg(association_rows=('Accession','size'),unique_accessions=('Accession','nunique'),genes=('Gene','nunique')).sort_values('association_rows',ascending=False)
        tech.to_csv(tables/f'{kind.lower()}_technique_summary.csv')
        bar(gene_counts.set_index('Gene').unique_accessions,f'Mouse {kind}: catalog coverage by gene','Unique accession identifiers',fig/f'{kind.lower()}_top_genes_by_accession.png')
        bar(tech.association_rows,f'Mouse {kind}: associations by technique','Gene–dataset association rows',fig/f'{kind.lower()}_associations_by_technique.png')
    # Deduplicate accession/gene pairs; source can contain the same identifier.
    left=catalog['TF'][['Gene','Accession']].drop_duplicates().rename(columns={'Gene':'TF'})
    right=catalog['RBP'][['Gene','Accession']].drop_duplicates().rename(columns={'Gene':'RBP'})
    shared=left.merge(right,on='Accession',how='inner')
    shared.sort_values(['TF','RBP','Accession']).to_csv(tables/'tf_rbp_shared_accessions.csv',index=False)
    tfsize=left.groupby('TF').Accession.nunique(); rbpsize=right.groupby('RBP').Accession.nunique()
    pairs=shared.groupby(['TF','RBP']).Accession.nunique().rename('shared_accessions').reset_index()
    pairs['tf_accessions']=pairs.TF.map(tfsize); pairs['rbp_accessions']=pairs.RBP.map(rbpsize)
    pairs['jaccard_similarity']=pairs.shared_accessions/(pairs.tf_accessions+pairs.rbp_accessions-pairs.shared_accessions)
    pairs.sort_values(['shared_accessions','jaccard_similarity'],ascending=False).to_csv(tables/'tf_rbp_shared_accession_statistics.csv',index=False)
    if not pairs.empty:
        top_tf=pairs.groupby('TF').shared_accessions.sum().nlargest(20).index
        top_rbp=pairs.groupby('RBP').shared_accessions.sum().nlargest(20).index
        matrix=pairs.pivot(index='TF',columns='RBP',values='shared_accessions').reindex(index=top_tf,columns=top_rbp).fillna(0)
        matrix.to_csv(tables/'top_tf_rbp_shared_accession_matrix.csv')
        f,ax=plt.subplots(figsize=(max(9,len(top_rbp)*.55),max(6,len(top_tf)*.4)))
        im=ax.imshow(matrix,aspect='auto',cmap='Blues'); ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=60,ha='right'); ax.set_yticks(range(len(matrix.index)),matrix.index)
        ax.set_title('Mouse catalog: shared accession identifiers (top genes)'); ax.set_xlabel('RBP'); ax.set_ylabel('TF'); f.colorbar(im,ax=ax,label='Shared accessions'); f.tight_layout(); f.savefig(fig/'tf_rbp_shared_accession_heatmap.png',dpi=200,bbox_inches='tight'); plt.close(f)
    (base/'README.md').write_text('''# Mouse catalog analysis\n\nThe mouse inputs are catalogs of gene–dataset associations, not alternative-splicing event results. Therefore exact splicing-event overlap, age persistence, and sex-differential splicing cannot be computed from these files. Figures summarize catalog coverage and exact shared accession identifiers instead. A shared accession means the same identifier appears beside both genes; it does not establish a shared splicing event, direct regulation, or even that either gene was experimentally perturbed.\n\nAll technique sheets, including split and unclassified sheets, are included; summary/readme sheets are excluded. Catalog rows with missing gene or accession are excluded. The top-gene charts count distinct accession strings per gene; technique charts count association rows, not unique studies. Pair statistics deduplicate gene/accession pairs and use exact accession strings; different identifier types or aliases are not reconciled. Only nonzero pairs are in the statistics table; the heatmap shows up to 20 genes per class ranked by total pairwise overlap. Entries, gene associations, tissues and identifiers require source-level verification. Tables are in `tables/`.\n''')
    print(f'Mouse: {len(catalog["TF"]):,} TF rows, {len(catalog["RBP"]):,} RBP rows, {len(pairs):,} nonzero shared-accession pairs',flush=True)

if __name__ == '__main__':
    niranjan()
    mouse()
