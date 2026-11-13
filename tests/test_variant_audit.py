import numpy as np
import pytest

from src.variant_audit import align_images,annotations_agree,orient_boxes


def test_rotated_box_coordinates():
    assert orient_boxes([[0,.2,.3,.1,.4]],1)[0]==pytest.approx([0,.3,.8,.4,.1])
    assert orient_boxes([[0,.2,.3,.1,.4]],2)[0]==pytest.approx([0,.8,.7,.1,.4])
    assert orient_boxes([[0,.2,.3,.1,.4]],1,True)[0]==pytest.approx([0,.7,.8,.4,.1])


def test_all_eight_image_alignments_recovered():
    image = np.random.default_rng(42).integers(0,255,(64,64,3),dtype=np.uint8)
    for k in range(4):
        for flip in [False,True]:
            other = np.rot90(image,k)
            if flip:
                other = np.fliplr(other)
            mae,_,_,aligned = align_images(image,other)
            assert mae==0 and np.array_equal(image,aligned)


def test_annotation_matching_is_order_independent():
    a = [[0,.2,.3,.1,.1],[0,.8,.8,.1,.1]]
    assert annotations_agree(a,list(reversed(a)))[0]
    assert not annotations_agree(a,a[:1])[0]
    assert not annotations_agree(a,[[0,.5,.5,.1,.1],a[1]])[0]


def test_annotation_match_allows_small_localization_variation():
    assert annotations_agree([[0,.5,.5,.2,.2]],[[0,.51,.5,.2,.2]])[0]


@pytest.mark.parametrize('conflict',[False,True])
def test_orientation_audit_quarantines_conflict_family(tmp_path,conflict):
    import io,zipfile
    from PIL import Image
    from src.variant_audit import investigate_variants
    image = np.random.default_rng(42).integers(0,255,(64,64,3),dtype=np.uint8)
    source_paths = {}
    for source,pixels in [('a',image),('b',np.rot90(image,1))]:
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream,format='PNG')
        path = tmp_path/(source+'.zip')
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('dataset/train/shared.png',stream.getvalue())
        source_paths[source] = path
    a = [[0,.2,.3,.1,.4]]
    b = orient_boxes(a,1)
    if conflict:
        b += [[0,.7,.7,.1,.1]]
    records = [dict(id=s,source=s,member='dataset/train/shared.png',normalized_source_name='shared',
                    boxes=boxes,reasons=[],status='accepted') for s,boxes in [('a',a),('b',b)]]
    config = dict(cache=tmp_path/'cache/ingestion.jsonl',sources=source_paths,workers=2,phash_distance=4)
    evidence,result = investigate_variants(records,config,{'test':'immutable PNG fixtures'})
    assert result['near_identical_aligned_pairs']==1
    assert bool(evidence)==conflict
    assert all(r['status']==('quarantined' if conflict else 'accepted') for r in records)
    # Reusing thumbnail descriptors must reproduce the same decision.
    again,counts = investigate_variants(records,config,{'test':'immutable PNG fixtures'})
    assert again==evidence and counts==result


def test_variant_audit_without_candidate_groups(tmp_path):
    from src.variant_audit import investigate_variants
    evidence,result = investigate_variants([],dict(phash_distance=4),{})
    assert not evidence and result['pairs_compared']==0
