"""Verify quarantine coverage, raw preservation and final artifact completeness."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from .prepare_dataset import ROOT, file_hash, load_config, load_manifest, write_json


def verify(config):
    records = load_manifest(config)
    report = config['reports']
    summary = json.loads((report/'summary.json').read_text())
    validation = json.loads((report/'validation.json').read_text())
    assert validation['ultralytics']['status']=='passed'
    assert Counter(r['status'] for r in records)==Counter(summary['statuses'])
    assert len(records)==summary['original_images']
    assert sum(summary['statuses'].values())==summary['original_images']
    assert sum(v['images'] for v in validation['splits'].values())==summary['final_images']
    assert sum(v['objects'] for v in validation['splits'].values())==summary['final_objects']
    for source,meta in summary['source_archives'].items():
        assert file_hash(config['sources'][source])==meta['sha256']
    baseline = report/'preservation_baseline.json'
    preservation = {}
    if baseline.exists():
        for name,digest in json.loads(baseline.read_text()).items():
            path = ROOT/'data'/name if name=='ship1.JPG' else ROOT/name
            assert file_hash(path)==digest, f'Original changed: {path}'
            preservation[name] = 'unchanged'
    index = json.loads((config['quarantine']/'index.json').read_text())
    rejected = [r for r in records if r['status']=='quarantined']
    assert {r['id'] for r in index}=={r['id'] for r in rejected}
    expected_members = set()
    with zipfile.ZipFile(config['quarantine']/'quarantined.zip') as quarantine:
        assert quarantine.testzip() is None
        for source,path in config['sources'].items():
            with zipfile.ZipFile(path) as original:
                for record in rejected:
                    if record['source']!=source:
                        continue
                    member = record['quarantine_member']
                    expected_members.add(member)
                    assert quarantine.read(member)==original.read(record['member'])
                    if record['raw_annotation'] is not None:
                        label = str(Path(member).with_suffix('.txt')).replace('\\','/')
                        expected_members.add(label)
                        assert quarantine.read(label).decode('utf-8')==original.read(record['label_member']).decode('utf-8-sig')
                orphan_path = report/'manifests/orphans.jsonl'
                for line in orphan_path.read_text().splitlines():
                    orphan = json.loads(line)
                    if orphan['source']==source:
                        member = source+'/orphan_labels/'+Path(orphan['member']).name
                        expected_members.add(member)
                        assert quarantine.read(member)==original.read(orphan['member'])
        assert set(quarantine.namelist())==expected_members
    # Every original polygon, including rejected rows, is preserved verbatim.
    originals = {r['id']:[a['original'] for a in r['annotation_audit'] if a['format']=='polygon']
                 for r in records if 'polygon' in r['annotation_format']}
    polygon_rows = [json.loads(line) for line in (report/'manifests/original_polygons.jsonl').read_text().splitlines()]
    assert {r['id']:[row['original'] for row in r['rows']] for r in polygon_rows}==originals
    variant_pairs = [json.loads(line) for line in (report/'manifests/variant_conflicts.jsonl').read_text().splitlines()]
    variant_names = {pair['normalized_source_name'] for pair in variant_pairs}
    assert len(variant_pairs)==summary['relations']['transformed_variant_audit']['conflicting_pairs']
    assert all(r['status']=='quarantined' for r in records if r['normalized_source_name'] in variant_names)
    import xml.etree.ElementTree as ET
    test_file = report/'test_results.xml'
    tests = None
    if test_file.exists():
        suites = ET.parse(test_file).getroot().findall('testsuite')
        assert all(int(s.attrib.get('errors',0))==0 and int(s.attrib.get('failures',0))==0 for s in suites)
        tests = sum(int(s.attrib['tests']) for s in suites)
    # Ultralytics must not have silently repaired/recompressed any exported JPEG.
    from concurrent.futures import ThreadPoolExecutor
    def unchanged(record):
        return file_hash(config['output']/record['output_image'])==record['file_sha256']
    accepted = [r for r in records if r['status']=='accepted']
    assert all(not r.get('review_flags') for r in accepted)
    approved_clips = set(config.get('approved_clipped_image_ids',[]))
    assert all(not r['clipped_boxes'] or r['id'] in approved_clips for r in accepted)
    assert all(min(b[3]*r['width'],b[4]*r['height'])>=1-1e-6 for r in accepted for b in r['boxes'])
    lookup = {r['id']:r for r in records}
    assert all(lookup[r['duplicate_of']]['status']=='accepted' for r in records if r['status']=='duplicate_removed')
    with ThreadPoolExecutor(max_workers=config['workers']) as executor:
        assert all(executor.map(unchanged,accepted)), 'Loader changed exported image bytes'
    figures = sorted((report/'figures').glob('*.png'))
    assert len(figures)>=5 and all(p.stat().st_size>1000 for p in figures)
    assert (report/'dataset_preparation.md').exists()
    result = {'status':'passed','source_archives_unchanged':True,'original_files':preservation,
              'quarantine_images_verified':len(rejected),'quarantine_zip_members':len(expected_members),
              'original_polygon_rows_verified':sum(map(len,originals.values())),
              'flagged_variant_families_fully_quarantined':len(variant_names),
              'automated_tests_passed':tests,
              'export_hashes_after_ultralytics':'passed','figures':len(figures),
              'all_input_images_accounted_for':True}
    write_json(report/'artifact_validation.json',result)
    print(json.dumps(result,indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/dataset.yaml')
    args = parser.parse_args()
    verify(load_config(args.config))


if __name__=='__main__':
    main()
