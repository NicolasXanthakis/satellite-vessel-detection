"""Orientation-aware investigation of same-source image variants.

Near-identical image content is evidence of an annotation conflict, not a reason
to select one label set. The complete source-name family is quarantined.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .dataset_core import phash


def orient_boxes(boxes, turns=0, flip=False):
    output = []
    for c,x,y,w,h in boxes:
        for _ in range(turns%4):
            x,y,w,h = y,1-x,h,w
        if flip:
            x = 1-x
        output.append([c,x,y,w,h])
    return output


def annotations_agree(a,b,minimum_iou=.5):
    if len(a)!=len(b):
        return False,0.
    if not a:
        return True,1.
    a,b = np.asarray(a,dtype=float)[:,1:],np.asarray(b,dtype=float)[:,1:]
    a1,a2 = a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2
    b1,b2 = b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2
    intersection = np.maximum(0,np.minimum(a2[:,None,:],b2[None,:,:])-np.maximum(a1[:,None,:],b1[None,:,:])).prod(axis=2)
    area_a,area_b = a[:,2:].prod(axis=1),b[:,2:].prod(axis=1)
    iou = intersection/(area_a[:,None]+area_b[None,:]-intersection+1e-15)
    rows,cols = linear_sum_assignment(-iou)
    lowest = float(iou[rows,cols].min())
    return lowest>=minimum_iou,lowest


def align_images(reference,candidate):
    """Find the best of eight right-angle rotation/reflection alignments."""
    best = None
    for turns in range(4):
        rotated = np.rot90(candidate,turns)
        for flip in [False,True]:
            oriented = np.fliplr(rotated) if flip else rotated
            mae = float(np.abs(reference.astype(np.int16)-oriented.astype(np.int16)).mean())/255
            if best is None or mae<best[0]:
                best = (mae,turns,flip,oriented)
    return best


def thumbnail(job):
    record,image_bytes = job
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert('RGB').resize((64,64),Image.Resampling.LANCZOS)
        return record['id'],np.asarray(rgb,dtype=np.uint8)


def investigate_variants(records,config,fingerprints):
    groups = defaultdict(list)
    for record in records:
        if record['boxes'] and not record.get('annotation_errors') and not record.get('decode_error'):
            groups[record['normalized_source_name']].append(record)
    groups = {name:rows for name,rows in groups.items() if len(rows)>1}
    needed = {r['id'] for rows in groups.values() for r in rows}
    if not needed:
        return [], {'source_name_groups_investigated':0,'thumbnail_images':0,'pairs_compared':0,
                    'near_identical_aligned_pairs':0,'conflicting_pairs':0,
                    'conflicting_source_name_families':0,'images_in_conflicting_families':0,
                    'maximum_thumbnail_mae':config.get('variant_max_mae',4/255),
                    'minimum_box_iou':config.get('variant_min_box_iou',.5),'phash_radius':config['phash_distance']}
    cache = config['cache'].parent/'variant_thumbnails.npz'
    meta = cache.with_suffix('.meta.json')
    identity = {'version':1,'sources':fingerprints,'pillow':version('Pillow'),
                'ids_sha256':hashlib.sha256('\n'.join(sorted(needed)).encode()).hexdigest()}
    if cache.exists() and meta.exists() and json.loads(meta.read_text())==identity:
        with np.load(cache,allow_pickle=False) as values:
            pixels = values['pixels']
            descriptors = dict(zip(values['ids'].tolist(),pixels))
        print('Loaded orientation-audit thumbnail cache',flush=True)
    else:
        descriptors = {}
        with ThreadPoolExecutor(max_workers=config['workers']) as executor:
            for source,path in config['sources'].items():
                subset = [r for r in records if r['source']==source and r['id'] in needed]
                with zipfile.ZipFile(path) as archive:
                    for start in range(0,len(subset),64):
                        jobs = [(r,archive.read(r['member'])) for r in subset[start:start+64]]
                        descriptors.update(executor.map(thumbnail,jobs))
                        if start%2048==0:
                            print(f'Orientation audit thumbnails {source}: {min(start+64,len(subset))}/{len(subset)}',flush=True)
        ids = sorted(descriptors)
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,ids=np.asarray(ids),pixels=np.stack([descriptors[key] for key in ids]))
        meta.write_text(json.dumps(identity,indent=2)+'\n')
    evidence,conflict_names = [],set()
    compared = matched = 0
    for name,rows in sorted(groups.items()):
        for index,a in enumerate(rows):
            reference = descriptors[a['id']]
            # Near-uniform thumbnails cannot establish reliable content identity.
            if float(reference.std())<2.:
                continue
            for b in rows[index+1:]:
                compared += 1
                mae,turns,flip,oriented = align_images(reference,descriptors[b['id']])
                if mae>config.get('variant_max_mae',4/255):
                    continue
                distance = (phash(Image.fromarray(reference))^phash(Image.fromarray(oriented))).bit_count()
                if distance>config['phash_distance']:
                    continue
                matched += 1
                agrees,minimum_iou = annotations_agree(a['boxes'],orient_boxes(b['boxes'],turns,flip),
                                                      config.get('variant_min_box_iou',.5))
                if not agrees:
                    conflict_names.add(name)
                    evidence.append({'a':a['id'],'b':b['id'],'normalized_source_name':name,
                                     'turns_counterclockwise':turns,'horizontal_flip_after_rotation':flip,
                                     'thumbnail_mean_absolute_error':mae,'phash_distance':distance,
                                     'object_counts':[len(a['boxes']),len(b['boxes'])],
                                     'minimum_matched_box_iou':minimum_iou,
                                     'reason':'object_count_disagreement' if len(a['boxes'])!=len(b['boxes']) else 'box_localization_disagreement'})
    affected = 0
    for record in records:
        if record['normalized_source_name'] in conflict_names:
            record['reasons'] = sorted(set(record['reasons']+['transformed_duplicate_annotation_conflict']))
            record['status'] = 'quarantined'
            affected += 1
    result = {'source_name_groups_investigated':len(groups),'thumbnail_images':len(needed),
              'pairs_compared':compared,'near_identical_aligned_pairs':matched,
              'conflicting_pairs':len(evidence),'conflicting_source_name_families':len(conflict_names),
              'images_in_conflicting_families':affected,'maximum_thumbnail_mae':config.get('variant_max_mae',4/255),
              'minimum_box_iou':config.get('variant_min_box_iou',.5),
              'phash_radius':config['phash_distance']}
    print('Orientation-aware annotation audit: '+json.dumps(result),flush=True)
    return evidence,result
