"""Archive-first preparation. Run with ``python -m src.prepare_dataset``."""
from __future__ import annotations

import argparse
import hashlib
import io
import inspect
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .dataset_core import (BKTree, UnionFind, annotation_signature, grouped_split,
                           normalize_source, parse_annotations, phash, scene_key, split_digest)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for key in ['output','quarantine','reports','cache']:
        config[key] = ROOT / config[key]
    config['sources'] = {key: ROOT / value for key,value in config['sources'].items()}
    # Never allow generated artifacts to target a source or a directory above it.
    for key in ['output','quarantine','reports','cache']:
        target = config[key].resolve()
        if not target.is_relative_to(ROOT) or target == ROOT:
            raise ValueError(f'{key} must be inside the project: {target}')
        for source in config['sources'].values():
            if source.resolve().is_relative_to(target):
                raise ValueError(f'{key} overlaps a raw source')
    return config


def inspect_record(job):
    record, image_bytes = job
    record['file_sha256'] = hashlib.sha256(image_bytes).hexdigest()
    record['reasons'] = []
    record['boxes'], record['annotation_audit'], errors = parse_annotations(record['raw_annotation'] or '')
    if record['raw_annotation'] is None:
        record['reasons'].append('missing_annotation')
    elif not record['raw_annotation'].strip():
        record['reasons'].append('unverified_empty_annotation')
    if errors:
        record['reasons'].append('invalid_annotation')
        record['annotation_errors'] = errors
    kinds = sorted({row['format'] for row in record['annotation_audit']})
    record['annotation_format'] = '+'.join(kinds) if kinds else 'missing' if record['raw_annotation'] is None else 'empty'
    record['original_objects'] = len(record['annotation_audit'])
    record['converted_polygons'] = sum(row['format']=='polygon' and 'box' in row for row in record['annotation_audit'])
    record['clipped_boxes'] = sum(row.get('clipped',False) for row in record['annotation_audit'])
    record['collapsed_object_rows'] = sum('box' in row for row in record['annotation_audit']) - len(record['boxes'])
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            record['width'], record['height'] = image.size
            record['exif_orientation'] = image.getexif().get(274, 1)
            if record['exif_orientation'] not in (None, 1):
                record['reasons'].append('unresolved_exif_orientation')
            rgb = image.convert('RGB')
            record['pixel_sha256'] = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            record['phash'] = f'{phash(rgb):016x}'
    except Exception as exc:
        record['reasons'].append('image_decode_error')
        record['decode_error'] = str(exc)
        record['width'] = record['height'] = 0
        record['pixel_sha256'] = record['phash'] = None
    record['normalized_source_name'] = normalize_source(Path(record['member']).stem)
    record['parent_scene'] = scene_key(record['normalized_source_name'])
    return record


def ingest(config):
    fingerprints = {key: {'sha256': file_hash(path), 'bytes': path.stat().st_size,
                          'path': path.relative_to(ROOT).as_posix()} for key,path in config['sources'].items()}
    # Only ingestion/inspection logic affects this cache. Export/report edits do not.
    inspection_functions = [ingest, inspect_record, parse_annotations, normalize_source, scene_key, phash]
    code_hash = hashlib.sha256('\n'.join(inspect.getsource(f) for f in inspection_functions).encode()).hexdigest()
    expected = {'schema': SCHEMA, 'sources': fingerprints, 'code_sha256': code_hash,
                'pillow': version('Pillow'), 'numpy': version('numpy'), 'scipy': version('scipy')}
    cache = config['cache']
    meta = cache.with_suffix('.meta.json')
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == expected:
        print('Loading validated ingestion cache', flush=True)
        rows = [json.loads(line) for line in cache.read_text(encoding='utf-8').splitlines()]
        orphan_path = cache.with_suffix('.orphans.json')
        return rows, json.loads(orphan_path.read_text()), fingerprints
    records, orphans = [], []
    with ThreadPoolExecutor(max_workers=config['workers']) as executor:
        for source, path in sorted(config['sources'].items()):
            with zipfile.ZipFile(path) as archive:
                members = {n for n in archive.namelist() if not n.endswith('/')}
                images = sorted(n for n in members if Path(n).suffix.lower() in {'.jpg','.jpeg','.png'})
                labels = {n for n in members if n.lower().endswith('.txt')}
                used = set()
                for start in range(0, len(images), 64):
                    jobs = []
                    for member in images[start:start+64]:
                        label = str(Path(member).with_suffix('.txt')).replace('\\','/')
                        if '/images/' in member:
                            label = label.replace('/images/', '/labels/')
                        if label in labels:
                            used.add(label)
                            raw = archive.read(label).decode('utf-8-sig')
                        else:
                            raw = None
                        record = {'id': source+'_'+hashlib.sha256(member.encode()).hexdigest()[:20],
                                  'source': source, 'original_split': member.split('/')[1],
                                  'filename': Path(member).name, 'member': member,
                                  'label_member': label if label in labels else None,
                                  'raw_annotation': raw}
                        jobs.append((record, archive.read(member)))
                    records.extend(executor.map(inspect_record, jobs))
                    if start % 1024 == 0:
                        print(f'Ingest {source}: {min(start+64,len(images))}/{len(images)}', flush=True)
                for label in sorted(labels-used):
                    orphans.append({'source':source,'member':label,'reason':'orphan_annotation',
                                    'raw_annotation':archive.read(label).decode('utf-8-sig')})
    records.sort(key=lambda r:r['id'])
    write_jsonl(cache, records)
    write_json(meta, expected)
    write_json(cache.with_suffix('.orphans.json'), orphans)
    return records, orphans, fingerprints


def relate_and_deduplicate(records, config):
    union = UnionFind(len(records))
    indexes = {key:defaultdict(list) for key in ['file_sha256','pixel_sha256','normalized_source_name','parent_scene']}
    edges = []
    for index,record in enumerate(records):
        for key, groups in indexes.items():
            value = record.get(key)
            if value and (key != 'parent_scene' or config['group_parent_scenes']):
                if value in groups:
                    other = groups[value][0]
                    union.union(index,other)
                    edges.append({'a':records[other]['id'],'b':record['id'],'kind':key})
                groups[value].append(index)
    # Perceptual matches are grouping evidence only, never an automatic deletion.
    tree = BKTree()
    for index,record in enumerate(records):
        if record['phash'] is None:
            continue
        value = int(record['phash'],16)
        for other,distance in tree.query(value,config['phash_distance']):
            # Same-source names and hashes already have recorded evidence.
            if union.find(index) != union.find(other):
                edges.append({'a':records[other]['id'],'b':record['id'],'kind':'perceptual',
                              'distance':distance})
            union.union(index,other)
        tree.add(value,index)
    # Quarantine every exact-image cluster with unresolved label disagreement.
    exact = UnionFind(len(records))
    for key in ['file_sha256','pixel_sha256']:
        for group in indexes[key].values():
            for index in group[1:]:
                exact.union(group[0],index)
    clusters = defaultdict(list)
    for index in range(len(records)):
        clusters[exact.find(index)].append(index)
    conflict_names = set()
    duplicates = []
    conflicts = []
    for cluster in clusters.values():
        if len(cluster) < 2:
            continue
        canonical = records[cluster[0]]
        unresolved = any(records[i]['reasons'] for i in cluster)
        disagreement = any(annotation_signature(records[i]['boxes']) != annotation_signature(canonical['boxes']) for i in cluster)
        if unresolved or disagreement:
            conflicts.append([records[i]['id'] for i in cluster])
            for index in cluster:
                records[index]['reasons'].append('exact_duplicate_annotation_conflict')
                conflict_names.add(records[index]['normalized_source_name'])
        else:
            # Stable canonical selection, independent of former split.
            for index in cluster[1:]:
                records[index]['duplicate_of'] = canonical['id']
                duplicates.append(records[index]['id'])
    for record in records:
        if record['normalized_source_name'] in conflict_names and 'exact_duplicate_annotation_conflict' not in record['reasons']:
            record['reasons'].append('related_annotation_conflict')
    members = defaultdict(list)
    for index,record in enumerate(records):
        members[union.find(index)].append(record['id'])
    group_ids = {key:'g_'+hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()[:20]
                 for key,ids in members.items()}
    for index,record in enumerate(records):
        record['group_id'] = group_ids[union.find(index)]
        record['status'] = 'quarantined' if record['reasons'] else 'duplicate_removed' if 'duplicate_of' in record else 'accepted'
        record['reasons'] = sorted(set(record['reasons']))
    return edges, {'exact_duplicate_clusters':sum(len(c)>1 for c in clusters.values()),
                   'conflict_clusters':conflicts,'removed_duplicate_ids':duplicates,
                   'relation_edges':dict(Counter(e['kind'] for e in edges)),
                   'all_groups':len(members),
                   'cross_dataset_normalized_source_groups':sum(
                       len({records[i]['source'] for i in group})>1
                       for group in indexes['normalized_source_name'].values())}


def serialize_boxes(boxes):
    return ''.join('0 '+' '.join(f'{v:.10f}' for v in row[1:])+'\n' for row in boxes)


def publish_staging(stage, output):
    """Windows antivirus/indexing can briefly deny directory renames."""
    for attempt in range(6):
        try:
            stage.rename(output)
            return
        except PermissionError:
            if attempt==5:
                raise
            time.sleep(min(2**attempt,5))


def export_dataset(records, orphans, config):
    output = config['output']
    if output.exists():
        raise FileExistsError(f'{output} already exists; use --validate-only or configure a new output')
    stage = output.with_name(output.name+'.staging')
    if stage.exists():
        raise FileExistsError(f'Unfinished staging folder exists: {stage}; inspect it before retrying')
    quarantine = config['quarantine']
    if (quarantine/'quarantined.zip').exists():
        raise FileExistsError('Quarantine ZIP already exists; use a new configured quarantine location')
    for split in ['train','val','test']:
        (stage/'images'/split).mkdir(parents=True,exist_ok=True)
        (stage/'labels'/split).mkdir(parents=True,exist_ok=True)
    quarantine.mkdir(parents=True,exist_ok=True)
    # Source JPEG bytes remain unchanged. No resizing or augmentation is applied.
    def write_accepted(record, image_bytes):
        destination = record['id']+Path(record['member']).suffix.lower()
        (stage/'images'/record['split']/destination).write_bytes(image_bytes)
        (stage/'labels'/record['split']/(record['id']+'.txt')).write_text(serialize_boxes(record['boxes']),encoding='utf-8')

    with ThreadPoolExecutor(max_workers=config['workers']) as executor, \
            zipfile.ZipFile(quarantine/'quarantined.zip','x',compression=zipfile.ZIP_STORED) as rejected:
        pending = []
        for source,path in sorted(config['sources'].items()):
            with zipfile.ZipFile(path) as archive:
                subset = [r for r in records if r['source']==source]
                for index,record in enumerate(subset):
                    if record['status']=='accepted':
                        destination = record['id']+Path(record['member']).suffix.lower()
                        record['output_image'] = f'images/{record["split"]}/{destination}'
                        pending.append(executor.submit(write_accepted,record,archive.read(record['member'])))
                        if len(pending)>=64:
                            for future in pending:
                                future.result()
                            pending.clear()
                    elif record['status']=='quarantined':
                        base = source+'/'+record['id']
                        rejected.writestr(base+Path(record['member']).suffix.lower(),archive.read(record['member']))
                        if record['raw_annotation'] is not None:
                            rejected.writestr(base+'.txt',record['raw_annotation'])
                        record['quarantine_member'] = base+Path(record['member']).suffix.lower()
                    if index % 2048 == 0:
                        print(f'Export {source}: {index}/{len(subset)}',flush=True)
                for orphan in orphans:
                    if orphan['source']==source:
                        rejected.writestr(source+'/orphan_labels/'+Path(orphan['member']).name,archive.read(orphan['member']))
        for future in pending:
            future.result()
    # Omit path so Ultralytics resolves paths relative to the YAML location.
    (stage/'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: ship\n',encoding='utf-8')
    publish_staging(stage,output)


def validate_export_record(record,output):
    path = output/record['output_image']
    image_bytes = path.read_bytes()
    assert hashlib.sha256(image_bytes).hexdigest()==record['file_sha256'], f'Changed export {path}'
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert('RGB')
        assert image.size==(record['width'],record['height'])
        assert hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()==record['pixel_sha256']
    text = (output/'labels'/record['split']/(record['id']+'.txt')).read_text()
    boxes,audit,annotation_errors = parse_annotations(text)
    assert not annotation_errors and len(boxes)==len(record['boxes']) and len(boxes)>0
    assert all(row['format']=='bbox' and not row['clipped'] for row in audit)
    # Decimal serialization can move exact rounding ties. Compare numerically,
    # rather than rounding to fewer decimal places and comparing strings.
    assert np.allclose(np.asarray(boxes),np.asarray(record['boxes']),atol=1e-9,rtol=0)
    return len(boxes)


def validate_export(records,config):
    output = config['output']
    accepted = [r for r in records if r['status']=='accepted']
    assignment = grouped_split(accepted,config['seed'],config['ratios'])
    expected = {r['id']:r['split'] for r in accepted}
    if assignment != expected:
        raise AssertionError('Recomputed split differs from saved manifest')
    if grouped_split(list(reversed(accepted)),config['seed'],config['ratios']) != expected:
        raise AssertionError('Splitting depends on input order')
    maps = {key:{} for key in ['file_sha256','pixel_sha256','group_id','normalized_source_name','parent_scene']}
    tree = BKTree()
    errors, split_stats = [], {}
    for split in ['train','val','test']:
        subset = [r for r in accepted if r['split']==split]
        images = {p.name for p in (output/'images'/split).iterdir() if p.is_file()}
        labels = {p.name for p in (output/'labels'/split).glob('*.txt')}
        wanted_images = {Path(r['output_image']).name for r in subset}
        wanted_labels = {r['id']+'.txt' for r in subset}
        assert images==wanted_images and labels==wanted_labels, f'Correspondence mismatch in {split}'
        with ThreadPoolExecutor(max_workers=config['workers']) as executor:
            objects = sum(executor.map(lambda r:validate_export_record(r,output),subset))
        for record in subset:
            for key,seen in maps.items():
                value = record[key]
                if value is None or (key=='parent_scene' and not config['group_parent_scenes']):
                    continue
                if value in seen:
                    if key in ['file_sha256','pixel_sha256'] or seen[value]!=split:
                        errors.append({'kind':key,'id':record['id'],'split':split,'other_split':seen[value]})
                seen[value]=split
            # Independently rescan perceptual relations across accepted splits.
            value = int(record['phash'],16)
            for other,_ in tree.query(value,config['phash_distance']):
                if other!=split:
                    errors.append({'kind':'perceptual','id':record['id'],'split':split,'other_split':other})
            tree.add(value,split)
        assert subset, f'Empty split: {split}'
        split_stats[split] = {'images':len(subset),'objects':objects,
                              'sources':dict(Counter(r['source'] for r in subset)),
                              'groups':len({r['group_id'] for r in subset})}
        print(f'Validated {split}: {len(subset)} images / {objects} objects',flush=True)
    assert not errors, f'Duplicate or group leakage: {errors[:5]}'
    return {'image_label_correspondence':'passed','geometry_classes_syntax':'passed',
            'all_exported_images_decoded':'passed','export_hash_integrity':'passed',
            'exact_duplicate_leakage':0,'related_group_leakage':0,
            'source_name_leakage':0,'recognized_parent_scene_leakage':0,'perceptual_radius_leakage':0,
            'reproducible_split':'passed','order_independent_split':'passed',
            'split_sha256':split_digest(expected),'splits':split_stats}


def validate_ultralytics(config):
    # Restrict settings/cache side effects to this project. Never instantiate a model.
    (ROOT/'.ultralytics'/'Ultralytics').mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    path = config['output']/'data.yaml'
    data = check_det_dataset(str(path),autodownload=False)
    result = {}
    for split in ['train','val','test']:
        dataset = YOLODataset(img_path=data[split], data=data, task='detect',
                              imgsz=640,augment=False,cache=False,batch_size=4)
        objects = sum(len(label['cls']) for label in dataset.labels)
        assert len(dataset)>0
        for index in sorted({0,len(dataset)//2,len(dataset)-1}):
            sample = dataset[index]
            assert sample['img'].ndim==3 and sample['bboxes'].shape[1]==4
        expected = [r for r in load_manifest(config) if r['status']=='accepted' and r['split']==split]
        assert len(dataset)==len(expected) and objects==sum(len(r['boxes']) for r in expected)
        # Verify every parsed label, not just aggregate counts: no silent dropping/reinterpretation.
        lookup = {r['id']:r for r in expected}
        for label in dataset.labels:
            record = lookup[Path(label['im_file']).stem]
            got = np.column_stack([label['cls'],label['bboxes']])
            label_path = config['output']/'labels'/split/(record['id']+'.txt')
            exported = np.asarray([list(map(float,line.split())) for line in label_path.read_text().splitlines()],dtype=np.float32)
            assert annotation_signature(got.tolist())==annotation_signature(exported.tolist())
        result[split] = {'images':len(dataset),'objects':objects,'sample_items_loaded':3,
                         'all_parsed_labels_match':True}
    return {'status':'passed','version':version('ultralytics'),'splits':result,'training_started':False}


def load_manifest(config):
    return [json.loads(line) for line in (config['reports']/'manifests/images.jsonl').read_text(encoding='utf-8').splitlines()]


def summarize(records,orphans,relations,config,fingerprints):
    accepted = [r for r in records if r['status']=='accepted']
    original = {}
    for source in sorted(config['sources']):
        original[source] = {}
        for split in sorted({r['original_split'] for r in records if r['source']==source}):
            rows = [r for r in records if r['source']==source and r['original_split']==split]
            orphan_rows = [r for r in orphans if r['source']==source and r['member'].split('/')[1]==split]
            orphan_objects = sum(len([line for line in r['raw_annotation'].splitlines() if line.strip()]) for r in orphan_rows)
            original[source][split] = {'images':len(rows),
                                      'label_files':sum(r['raw_annotation'] is not None for r in rows)+len(orphan_rows),
                                      'paired_label_files':sum(r['raw_annotation'] is not None for r in rows),
                                      'annotation_rows':sum(r['original_objects'] for r in rows)+orphan_objects,
                                      'orphan_label_files':len(orphan_rows)}
    return {'schema':SCHEMA,'seed':config['seed'],'target_ratios':config['ratios'],
            'source_archives':fingerprints,'original':original,
            'original_images':len(records),
            'original_annotation_rows':sum(v['annotation_rows'] for splits in original.values() for v in splits.values()),
            'final_images':len(accepted),'final_objects':sum(len(r['boxes']) for r in accepted),
            'statuses':dict(Counter(r['status'] for r in records)),
            'quarantine_reasons_nonexclusive':dict(Counter(reason for r in records for reason in r['reasons'])),
            'orphan_annotations':len(orphans),'excluded_standalone_images':['data/ship1.JPG'],
            'polygons_converted_in_accepted':sum(r['converted_polygons'] for r in accepted),
            'boxes_clipped_in_accepted':sum(r['clipped_boxes'] for r in accepted),
            'polygons_converted_during_inspection':sum(r['converted_polygons'] for r in records),
            'boxes_clipped_during_inspection':sum(r['clipped_boxes'] for r in records),
            'collapsed_object_rows_in_accepted':sum(r['collapsed_object_rows'] for r in accepted),
            'relations':relations,'accepted_groups':len({r['group_id'] for r in accepted}),
            'largest_accepted_group':max(Counter(r['group_id'] for r in accepted).values()),
            'environment':{'python':platform.python_version(),'packages':{name:version(name) for name in
                           ['numpy','Pillow','scipy','matplotlib','PyYAML','torch','ultralytics','pytest']}},
            'split_sha256':split_digest({r['id']:r['split'] for r in accepted})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/dataset.yaml')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true',help='Inspect, group and plan; do not export images')
    modes.add_argument('--validate-only',action='store_true',help='Read saved manifest and fully revalidate exports')
    modes.add_argument('--resume-export',action='store_true',help='Finalize and fully validate a completed/interrupted staging export without overwriting it')
    args = parser.parse_args()
    config = load_config(args.config)
    config['reports'].mkdir(parents=True,exist_ok=True)
    if args.validate_only:
        records = load_manifest(config)
        result = validate_export(records,config)
        result['ultralytics'] = validate_ultralytics(config)
        write_json(config['reports']/'validation.json',result)
        print(json.dumps(result,indent=2),flush=True)
        return
    records,orphans,fingerprints = ingest(config)
    edges,relations = relate_and_deduplicate(records,config)
    from .variant_audit import investigate_variants
    variant_evidence,variant_results = investigate_variants(records,config,fingerprints)
    relations['transformed_variant_audit'] = variant_results
    from .review_policy import apply_review_policy
    relations['review_policy'] = apply_review_policy(records,config)
    relations['removed_duplicate_ids'] = [r['id'] for r in records if r['status']=='duplicate_removed']
    accepted = [r for r in records if r['status']=='accepted']
    assignment = grouped_split(accepted,config['seed'],config['ratios'])
    assert assignment==grouped_split(list(reversed(accepted)),config['seed'],config['ratios'])
    assert set(assignment.values())=={'train','val','test'}
    for record in records:
        record['split'] = assignment.get(record['id'])
    summary = summarize(records,orphans,relations,config,fingerprints)
    if args.dry_run:
        summary['planned_splits'] = {s:dict(images=sum(r['split']==s for r in accepted),
                                          objects=sum(len(r['boxes']) for r in accepted if r['split']==s))
                                     for s in ['train','val','test']}
        write_json(config['reports']/'dry_run.json',summary)
        write_jsonl(config['reports']/'manifests/dry_run_images.jsonl',records)
        write_json(config['reports']/'manifests/dry_run_relations.json',relations)
        print(json.dumps(summary,indent=2),flush=True)
        return
    dry_run = config['reports']/'dry_run.json'
    if not dry_run.exists():
        raise RuntimeError('Run --dry-run before export')
    planned = json.loads(dry_run.read_text())
    assert planned['source_archives']==fingerprints and planned['split_sha256']==summary['split_sha256'], 'Dry run differs from actual plan'
    if args.resume_export:
        stage = config['output'].with_name(config['output'].name+'.staging')
        if stage.exists() and not config['output'].exists():
            publish_staging(stage,config['output'])
        if not config['output'].is_dir() or not (config['quarantine']/'quarantined.zip').is_file():
            raise RuntimeError('Resume requires an existing export and quarantine ZIP')
        for record in records:
            if record['status']=='accepted':
                destination = record['id']+Path(record['member']).suffix.lower()
                record['output_image'] = f'images/{record["split"]}/{destination}'
            elif record['status']=='quarantined':
                record['quarantine_member'] = record['source']+'/'+record['id']+Path(record['member']).suffix.lower()
        print('Resuming artifact creation and full validation; existing image/label files are not overwritten',flush=True)
    else:
        export_dataset(records,orphans,config)
    write_jsonl(config['reports']/'manifests/images.jsonl',records)
    write_jsonl(config['reports']/'manifests/orphans.jsonl',orphans)
    write_jsonl(config['reports']/'manifests/relations.jsonl',edges)
    write_jsonl(config['reports']/'manifests/variant_conflicts.jsonl',variant_evidence)
    write_jsonl(config['reports']/'manifests/review_decisions.jsonl',
                ({'id':r['id'],'source':r['source'],'member':r['member'],
                  'status':r['status'],'flags':r['review_flags']} for r in records if r['review_flags']))
    write_jsonl(config['reports']/'manifests/original_polygons.jsonl',
                ({'id':r['id'],'source':r['source'],'member':r['label_member'],
                  'status':r['status'],'rows':[row for row in r['annotation_audit'] if row['format']=='polygon']}
                 for r in records if 'polygon' in r['annotation_format']))
    write_json(config['quarantine']/'index.json',
               [{'id':r['id'],'source':r['source'],'member':r['member'],
                 'quarantine_member':r.get('quarantine_member'),'reasons':r['reasons']}
                for r in records if r['status']=='quarantined'])
    write_json(config['reports']/'summary.json',summary)
    result = validate_export(records,config)
    write_json(config['reports']/'validation.json',result)
    result['ultralytics'] = validate_ultralytics(config)
    write_json(config['reports']/'validation.json',result)
    from .report_dataset import generate_report
    generate_report(records,summary,result,config)
    assert all(file_hash(config['sources'][key])==value['sha256'] for key,value in fingerprints.items()), 'Original archive changed'
    print('Preparation, export, visualization and validation complete.',flush=True)
    print(json.dumps({'summary':str(config['reports']/'summary.json'),'validation':result},indent=2),flush=True)


if __name__=='__main__':
    main()
