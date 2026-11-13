"""Explicit, auditable review exclusions and source-pixel geometry policy."""
import re
from collections import Counter


def apply_review_policy(records,config):
    minimum = float(config.get('minimum_box_side_pixels',1.))
    if minimum<0:
        raise ValueError('Minimum pixel side cannot be negative')
    rules = config.get('reviewed_exclusions',[])
    approved_clips = set(config.get('approved_clipped_image_ids',[]))
    affected,subpixel_rows = 0,0
    counts = Counter()
    for record in records:
        flags = []
        tiny = []
        for index,box in enumerate(record['boxes']):
            width,height = box[3]*record['width'],box[4]*record['height']
            if min(width,height)<minimum-1e-6:
                tiny.append({'box_index':index,'width_pixels':width,'height_pixels':height})
        if tiny:
            flags.append({'reason':'subpixel_annotation','minimum_side_pixels':minimum,
                          'boxes':tiny})
            subpixel_rows += len(tiny)
        if config.get('require_clipped_annotation_review',True) and record.get('clipped_boxes',0) and record.get('id') not in approved_clips:
            flags.append({'reason':'clipped_annotation_requires_review',
                          'repaired_boxes':record['clipped_boxes'],
                          'rationale':'Clipping repairs geometry only. Reviewed examples were mislocalized; image-level annotation-alignment approval is required.',
                          'evidence':'review/clip_review.png'})
        for rule in rules:
            if record['source']==rule['source'] and re.search(rule['normalized_pattern'],record['normalized_source_name'],re.I):
                flags.append({'reason':rule['reason'],'rule_id':rule['id'],
                              'rationale':rule['rationale'],'evidence':rule['evidence']})
        record['review_flags'] = flags
        if flags:
            affected += 1
            reasons = {f['reason'] for f in flags}
            counts.update(reasons)
            record['reasons'] = sorted(set(record['reasons'])|reasons)
            record['status'] = 'quarantined'
    return {'minimum_box_side_pixels':minimum,'pixel_tolerance':1e-6,
            'require_clipped_annotation_review':config.get('require_clipped_annotation_review',True),
            'approved_clipped_image_ids':sorted(approved_clips),
            'affected_images':affected,'subpixel_box_rows':subpixel_rows,
            'reasons_nonexclusive':dict(counts),'rules':rules}
