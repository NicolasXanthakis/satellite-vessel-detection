from copy import deepcopy

from src.review_policy import apply_review_policy


def record(name='frame',source='ships',width=.1):
    return dict(source=source,normalized_source_name=name,width=100,height=100,
                boxes=[[0,.5,.5,width,.1]],reasons=[],status='accepted')


def config():
    return dict(minimum_box_side_pixels=1.,reviewed_exclusions=[dict(
        id='review',source='ships',normalized_pattern='frederikshavn|3b_visual',
        reason='unverified_semantic_annotation_family',rationale='reviewed evidence',evidence='review.png')])


def test_semantic_family_review_is_source_scoped_and_audited():
    rows = [record('frederikshavn1subset5'),record('2019_3b_visual_1'),record('frederikshavn1subset5','aerial'),record()]
    result = apply_review_policy(rows,config())
    assert [r['status'] for r in rows]==['quarantined','quarantined','accepted','accepted']
    assert result['affected_images']==2 and rows[0]['review_flags'][0]['evidence']=='review.png'


def test_subpixel_box_quarantines_entire_image_without_rewriting_labels():
    row = record(width=.009)
    row['boxes'].append([0,.3,.3,.2,.2])
    original = deepcopy(row['boxes'])
    result = apply_review_policy([row],config())
    assert row['status']=='quarantined' and row['boxes']==original
    assert result['subpixel_box_rows']==1


def test_one_pixel_floating_point_tolerance():
    row = record(width=.009999999999999998)
    apply_review_policy([row],config())
    assert row['status']=='accepted'


def test_overlapping_review_reasons_do_not_double_count_images():
    row = record('frederikshavn1subset5',width=.009)
    result = apply_review_policy([row],config())
    assert result['affected_images']==1 and len(row['reasons'])==2


def test_geometric_clipping_does_not_imply_annotation_approval():
    row = record()
    row.update(id='review_needed',clipped_boxes=1)
    result = apply_review_policy([row],config())
    assert row['status']=='quarantined'
    assert result['reasons_nonexclusive']['clipped_annotation_requires_review']==1


def test_explicit_clipped_image_approval_retains_otherwise_valid_image():
    row = record()
    row.update(id='reviewed',clipped_boxes=1)
    policy = config()
    policy['approved_clipped_image_ids'] = ['reviewed']
    apply_review_policy([row],policy)
    assert row['status']=='accepted'
