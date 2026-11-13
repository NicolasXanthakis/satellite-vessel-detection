"""Standalone statistical figures and traceable annotation examples."""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from PIL import Image


LIMITATIONS = """- Dataset origins, download URLs and licenses were not supplied in the archives and are unverified. Do not claim redistribution rights; obtain and record the original providers' terms before publishing imagery.
- This is a positive-only detection dataset: empty annotations were not verified as negatives and were excluded. Background performance and false-positive rates will require an independently annotated negative set.
- Filename normalization and 64-bit DCT perceptual hashes (Hamming radius 4) provide conservative grouping, not proof of common provenance. Perceptual matches can over-group visually similar water/ships and miss rotations, strong crops, color changes and other transforms. Transitive matches can produce large groups.
- Geographic acquisition IDs are grouped when recognizable in filenames. No reliable CRS, raster geotransforms, scene footprints or complete provenance are supplied. Spatial/temporal independence cannot be guaranteed for records without metadata, or nearby acquisitions with different IDs. This is not yet a georeferenced vessel-location product.
- The original train/validation/test designations are provenance only; all splits were rebuilt. Existing evaluation benchmarks are not preserved.
- Polygon-to-axis-aligned-box conversion loses shape/orientation information. Original polygon rows remain in the manifest, the polygon JSONL and the untouched archives.
- Clipping fixes geometric bounds but cannot prove the original localization was correct. Duplicate-image label disagreements show source annotation quality issues; a systematic human overlay review remains necessary.
- Requiring approval for clipped images and excluding unverified semantic families can remove correct examples. This reduces boundary-case/source coverage and introduces selection bias. No clipped-image approvals were available in this run.
- Sources contain varied image scales/resolutions and likely different sensors. No verified ground sampling distances are available. Object size in pixels is not vessel size in meters.
- Automated validity and loader checks establish technical usability, not detector performance. No model was trained and no accuracy result is claimed.
"""


def generate_report(records,summary,validation,config):
    directory = config['reports']
    figures = directory/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    accepted = [r for r in records if r['status']=='accepted']
    from .review_evidence import generate_review_evidence
    generate_review_evidence(records,config)
    from .dataset_core import features
    distributions = {}
    for split in ['train','val','test']:
        rows = [r for r in accepted if r['split']==split]
        distributions[split] = {
            'feature_counts':dict(Counter(f for r in rows for f in features(r))),
            'image_dimensions':dict(Counter(f'{r["width"]}x{r["height"]}' for r in rows)),
            'objects_per_image':dict(sorted(Counter(len(r['boxes']) for r in rows).items()))}
    (directory/'distributions.json').write_text(json.dumps(distributions,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':120,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titlesize':12,'font.size':10})
    colors = {'aerial':'#1b728a','ships':'#d68538'}

    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(figures/name,dpi=160,bbox_inches='tight')
        plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    sources = sorted(colors)
    x = np.arange(len(sources))
    before = [sum(r['source']==s for r in records) for s in sources]
    after = [sum(r['source']==s for r in accepted) for s in sources]
    bars_before = axes[0].bar(x-.18,before,.36,label='Original',color='#b7c6cf')
    bars_after = axes[0].bar(x+.18,after,.36,label='Accepted',color='#1b728a')
    axes[0].bar_label(bars_before,labels=[f'{v:,}' for v in before],padding=3)
    axes[0].bar_label(bars_after,labels=[f'{v:,}' for v in after],padding=3)
    axes[0].set(xticks=x,xticklabels=sources,ylabel='Images',title='Images before and after cleaning')
    axes[0].legend()
    statuses = summary['statuses']
    status_colors = {'accepted':'#1b728a','quarantined':'#b55b54','duplicate_removed':'#c5a248'}
    status_bars = axes[1].bar(range(len(statuses)),list(statuses.values()),color=[status_colors[s] for s in statuses])
    axes[1].bar_label(status_bars,labels=[f'{v:,}' for v in statuses.values()],padding=3)
    axes[1].set(xticks=range(len(statuses)),xticklabels=[s.replace('_','\n') for s in statuses],ylabel='Images',title='Processing outcomes')
    finish(fig,'cleaning_distribution.png')

    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    bins = np.arange(0, max(len(r['boxes']) for r in accepted)+2)-.5
    for source in sources:
        counts = [len(r['boxes']) for r in accepted if r['source']==source]
        axes[0].hist(counts,bins=bins,histtype='step',label=source,color=colors[source],linewidth=1.6)
    axes[0].set(yscale='log',xlabel='Objects per accepted image',ylabel='Images (log)',title='Object density by source')
    axes[0].legend()
    minimum_area = min(b[3]*b[4] for r in accepted for b in r['boxes'])
    area_bins = np.geomspace(minimum_area*.9,1,60)
    for source in sources:
        fractions = [b[3]*b[4] for r in accepted if r['source']==source for b in r['boxes']]
        axes[1].hist(fractions,bins=area_bins,histtype='step',label=source,color=colors[source],linewidth=1.6)
    axes[1].set(xscale='log',yscale='log',xlabel='Box area / image area',ylabel='Objects (log)',title='Normalized bounding-box area')
    axes[1].legend()
    finish(fig,'objects_and_box_sizes.png')

    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    for source in sources:
        rows = [r for r in accepted if r['source']==source]
        dims = Counter((r['width'],r['height']) for r in rows)
        axes[0].scatter([d[0] for d in dims],[d[1] for d in dims],s=[10+12*np.log1p(n) for n in dims.values()],
                        alpha=.65,label=source,color=colors[source])
    axes[0].set(xlabel='Image width (px)',ylabel='Image height (px)',title='Accepted image dimensions (marker size = count)')
    axes[0].legend()
    bottom = np.zeros(3)
    for source in sources:
        values = [sum(r['source']==source and r['split']==s for r in accepted) for s in ['train','val','test']]
        axes[1].bar(['train','val','test'],values,bottom=bottom,label=source,color=colors[source])
        bottom += values
    axes[1].set(ylabel='Accepted images',title='Rebuilt grouped splits')
    axes[1].legend()
    finish(fig,'dimensions_and_splits.png')

    # Metadata for every visual example is saved next to the figures.
    examples = []
    archives = {source:zipfile.ZipFile(path) for source,path in config['sources'].items()}

    def draw(ax,record,original=False):
        with Image.open(io.BytesIO(archives[record['source']].read(record['member']))) as image:
            image = image.convert('RGB')
            image.thumbnail((900,900))
            ax.imshow(image,extent=(0,1,1,0))
        if original:
            for row in record['annotation_audit']:
                values = list(map(float,row['original'].split()))
                if row['format']=='bbox':
                    _,x,y,w,h = values
                    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fill=False,edgecolor='#ffcc33',linewidth=1.4,clip_on=False))
                elif row['format']=='polygon':
                    ax.add_patch(Polygon(np.array(values[1:]).reshape(-1,2),closed=True,fill=False,edgecolor='#ffcc33',linewidth=1.2))
        else:
            for _,x,y,w,h in record['boxes']:
                ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fill=False,edgecolor='#5fff7e',linewidth=1.2))
        ax.set(xlim=(0,1),ylim=(1,0),xticks=[],yticks=[])
        ax.set_title(f'{record["source"]} | {record["id"][-8:]} | {len(record["boxes"])} objects\n'+
                     ('Original annotations' if original else 'Exported YOLO boxes' if record['status']=='accepted' else 'Converted boxes | QUARANTINED'))

    fig,axes = plt.subplots(2,2,figsize=(11,10))
    for row,source in enumerate(sources):
        candidates = sorted([r for r in accepted if r['source']==source],key=lambda r:r['id'])
        picks = [candidates[len(candidates)//3],candidates[2*len(candidates)//3]]
        for col,record in enumerate(picks):
            draw(axes[row,col],record)
            examples.append({'figure':'representative_annotations.png','id':record['id'],'source_member':record['member']})
    finish(fig,'representative_annotations.png')

    fig,axes = plt.subplots(2,2,figsize=(11,10))
    for row,source in enumerate(sources):
        repaired = sorted([r for r in records if r['source']==source and r['clipped_boxes'] and r['boxes'] and not r.get('annotation_errors')],
                          key=lambda r:(-r['clipped_boxes'],r['id']))
        record = repaired[0]
        draw(axes[row,0],record,True)
        draw(axes[row,1],record)
        examples.append({'figure':'repaired_annotations.png','id':record['id'],'source_member':record['member'],
                         'clipped_boxes':record['clipped_boxes']})
    finish(fig,'repaired_annotations.png')
    # This is evidence of geometry-only repair, not approval of label semantics.
    import shutil
    review_directory = directory/'review'
    review_directory.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(figures/'repaired_annotations.png',review_directory/'clip_review.png')
    # Show candidate cross-dataset variants, with traceable source IDs.
    related = {}
    for record in records:
        if record['boxes'] and not record.get('annotation_errors'):
            related.setdefault(record['normalized_source_name'],{}).setdefault(record['source'],record)
    shared = [(name,rows) for name,rows in sorted(related.items()) if len(rows)>1][:2]
    if shared:
        fig,axes = plt.subplots(len(shared),2,figsize=(11,5*len(shared)),squeeze=False)
        for row,(name,variants) in enumerate(shared):
            for col,source in enumerate(sorted(variants)):
                record = variants[source]
                draw(axes[row,col],record)
                examples.append({'figure':'overlap_candidates.png','id':record['id'],
                                 'source_member':record['member'],'normalized_source_name':name,
                                 'group_id':record['group_id'],'split':record['split']})
        finish(fig,'overlap_candidates.png')
    for archive in archives.values():
        archive.close()
    (directory/'visualization_examples.json').write_text(json.dumps(examples,indent=2)+'\n',encoding='utf-8')

    split_rows = '\n'.join(f'| {s} | {v["images"]:,} | {v["objects"]:,} | {v["sources"].get("aerial",0):,} | {v["sources"].get("ships",0):,} | {v["groups"]:,} |'
                           for s,v in validation['splits'].items())
    reasons = '\n'.join(f'| {reason} | {n:,} |' for reason,n in summary['quarantine_reasons_nonexclusive'].items())
    original_rows = '\n'.join(f'| {source} | {split} | {v["images"]:,} | {v["label_files"]:,} | {v["annotation_rows"]:,} |'
                              for source,splits in summary['original'].items() for split,v in splits.items())
    variant = summary['relations']['transformed_variant_audit']
    review = summary['relations']['review_policy']
    report = f'''# Dataset preparation report

This report describes an executed preparation run, not a model evaluation. Seed: **42**. Target split: **80/10/10**, with whole related-image groups taking precedence.

## Input inventory

| Source | Original split | Images | Label files | Object rows |
|---|---|---:|---:|---:|
{original_rows}

Original dataset images: **{summary['original_images']:,}**. Original object rows: **{summary['original_annotation_rows']:,}**, including orphan-label rows. The Ships extracted copy is not ingested. `data/ship1.JPG` has no verified annotation/provenance and is excluded. Source URLs and licenses are unverified.

## Cleaning decisions and outcomes

- Accepted: **{summary['final_images']:,} images**, **{summary['final_objects']:,} objects**.
- Quarantined: **{summary['statuses'].get('quarantined',0):,} images** in `data/quarantine/quarantined.zip`, with original labels where present and a separate index.
- Removed redundant exact duplicates: **{summary['statuses'].get('duplicate_removed',0):,} images**. Canonical copies are recorded in the manifest; originals remain in source archives.
- Orphan annotations excluded from export: **{summary['orphan_annotations']}**, retained in quarantine and the orphan manifest.
- Polygons converted and retained in the final dataset: **{summary['polygons_converted_in_accepted']:,}**; successfully converted during inspection: **{summary['polygons_converted_during_inspection']:,}**.
- Bounding boxes clipped and retained: **{summary['boxes_clipped_in_accepted']:,}**; clipped during inspection: **{summary['boxes_clipped_during_inspection']:,}**. Clipping uses corner-space intersection with the image.
- Duplicate converted object rows collapsed in accepted labels: **{summary['collapsed_object_rows_in_accepted']}**.
- Missing labels and unverified empty labels are never treated as negatives. Any unresolved row invalidates the entire image. Nonidentity EXIF orientation is quarantined rather than guessing coordinate conventions.
- Exact-image annotation disagreements are quarantined along with same-source-name variants. No conflicting label set is selected arbitrarily.
- Same-source variants are also compared under all eight right-angle rotation/reflection alignments. A 64x64 RGB thumbnail MAE <=4/255 and perceptual Hamming distance <=4 establish conservative near-identical-content evidence. Matched images with unequal object counts or a best one-to-one box matching IoU below 0.5 trigger quarantine of the entire source-name family. This is not an attempt to guess which annotation set is correct.
- Post-clipping boxes must have both sides >=1 source pixel (1e-6 pixel tolerance). Images with subpixel rows are quarantined as unresolved annotations, rather than silently dropping only their small objects.
- Border-crossing labels are geometrically clipped, but require explicit image-level annotation-alignment approval before export. Visual review found mislocalized examples whose semantics clipping could not fix. No clipped-image approvals were available, so repaired cases remain quarantined. The repair figure illustrates geometry conversion of rejected originals, not approved training labels.
- Explicit source-family review exclusions quarantine unverified class semantics identified in image overlays. These are documented policy decisions, not inferred class remappings. Rule definitions and per-image decisions are preserved in the configuration and `manifests/review_decisions.jsonl`.
- Original polygon text and per-row conversions/repairs are preserved in `manifests/original_polygons.jsonl` and the full manifest.

Reasons below are nonexclusive; their counts must not be summed as a unique image total.

| Quarantine reason | Images |
|---|---:|
{reasons}

## Final splits

| Split | Images | Objects | Aerial images | Ships images | Groups |
|---|---:|---:|---:|---:|---:|
{split_rows}

Accepted groups: **{summary['accepted_groups']:,}**; largest accepted group: **{summary['largest_accepted_group']:,} images**. Greedy allocation with deterministic refinement balances total images, objects, source, image size, aspect and object-density bins. No images are augmented, resized or recompressed during export.

`distributions.json` records per-split source/size/aspect/density bins, exact image dimensions and objects-per-image counts. Large indivisible groups can prevent exact balance in individual characteristics even when the overall 80/10/10 target is closely met.

## Leakage prevention and validation

- Exact duplicate clusters in the raw inputs: **{summary['relations']['exact_duplicate_clusters']}**; clusters with conflicting or unresolved annotations: **{len(summary['relations']['conflict_clusters'])}**. One Aerial pair lacks labels on both copies; the 19 Ships pairs have numeric annotation disagreements.
- Relations: `{json.dumps(summary['relations']['relation_edges'],sort_keys=True)}`. These are recorded union evidence edges, not an exhaustive count of all pairwise similarities.
- Normalized original filenames identify **{summary['relations']['cross_dataset_normalized_source_groups']}** candidate source groups shared between the two datasets. These are grouped globally, rather than relying on dataset-specific prefixes. No exact cross-dataset image match was found in the initial audit.
- Orientation-aware investigation compared **{variant['pairs_compared']:,} pairs** across **{variant['source_name_groups_investigated']:,} source-name groups**. It found **{variant['near_identical_aligned_pairs']:,} near-identical aligned pairs**, of which **{variant['conflicting_pairs']:,}** had incompatible annotations. This quarantined **{variant['images_in_conflicting_families']:,} images** in **{variant['conflicting_source_name_families']:,} families**. Pair-level evidence is in `manifests/variant_conflicts.jsonl`.
- Source-pixel review flagged **{review['subpixel_box_rows']:,} subpixel rows** across **{review['reasons_nonexclusive'].get('subpixel_annotation',0):,} input images**. Semantic-family review flagged **{review['reasons_nonexclusive'].get('unverified_semantic_annotation_family',0):,} input images**. These counts overlap other exclusion reasons. No such flags remain in the accepted export.
- Visual review showed boxes on clouds/non-vessel regions in named source families; matching families are held out pending verification. Some matching images contain correct vessel annotations, so this is a conservative exclusion. Exact rules and decisions remain in the configuration, `summary.json` and `manifests/review_decisions.jsonl`.
- Clipped images awaiting alignment approval: **{review['reasons_nonexclusive'].get('clipped_annotation_requires_review',0):,} input images**. Their repairs are retained in the row audit and review figures, while accepted images contain no unapproved repairs.
- Final exact duplicate leakage: **{validation['exact_duplicate_leakage']}**; final known related-group leakage: **{validation['related_group_leakage']}**.
- Independent accepted-image rescans also found zero normalized-source-name, recognized-parent-scene or perceptual-radius overlap across splits.
- All exported images decoded and matched their source file/pixel hashes. Every label was checked for positive area, in-image geometry, finite values, detection syntax and class ID 0.
- Split recomputation and reversed-input-order recomputation matched. Assignment SHA-256: `{validation['split_sha256']}`.
- Ultralytics **{validation['ultralytics']['version']}** loaded all three splits, matched every parsed annotation and loaded three transformed dataset items per split with augmentation disabled. No model was loaded or trained.
- Source archive SHA-256 values and package versions are recorded in `summary.json`; full source members, hashes, annotations, statuses and split assignments are in `manifests/images.jsonl`.

## Figures

![Cleaning](figures/cleaning_distribution.png)
![Object density and box sizes](figures/objects_and_box_sizes.png)
![Dimensions and splits](figures/dimensions_and_splits.png)
![Representative annotations](figures/representative_annotations.png)
![Clipping examples: original and repaired](figures/repaired_annotations.png)
![Candidate cross-dataset source variants](figures/overlap_candidates.png)

`visualization_examples.json` identifies the source image behind every example. Figure overlays are inspection artifacts; they do not modify dataset pixels.

![Source-family semantic review evidence](review/semantic_review.png)

`review/semantic_review_images.json` records the exact source members behind the review panel. The panel is regenerated from the raw archives by `src/review_evidence.py`.

## Limitations and next steps

{LIMITATIONS}

Obtain dataset licenses/provenance, conduct a human annotation review, add verified negatives and georeferenced imagery, then establish a fixed baseline training/evaluation protocol. Geospatial inference, tiling, vessel mapping and metric evaluation are planned stages, not implemented results.
'''
    (directory/'dataset_preparation.md').write_text(report,encoding='utf-8')


def main():
    from .prepare_dataset import ROOT,load_config,load_manifest
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/dataset.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    generate_report(load_manifest(config),json.loads((config['reports']/'summary.json').read_text()),
                    json.loads((config['reports']/'validation.json').read_text()),config)


if __name__=='__main__':
    main()
