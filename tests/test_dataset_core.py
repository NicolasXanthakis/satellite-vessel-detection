import math

import pytest

from src.dataset_core import (BKTree, UnionFind, annotation_signature, grouped_split,
                              normalize_source, parse_annotations, scene_key, split_digest)


def test_box_clipped_in_corner_space():
    boxes,audit,errors = parse_annotations('0 0.95 0.5 0.2 0.4')
    assert not errors
    assert boxes[0] == pytest.approx([0,.925,.5,.15,.4])
    assert audit[0]['clipped']


def test_polygon_enclosing_box_and_original_preserved():
    text = '0 0.1 0.2 0.5 0.2 0.3 0.8'
    boxes,audit,errors = parse_annotations(text)
    assert not errors
    assert boxes[0] == pytest.approx([0,.3,.5,.4,.6])
    assert audit[0]['original']==text and audit[0]['format']=='polygon'


@pytest.mark.parametrize('text',[
    '0 .5 .5 0 .2', '0 .5 .5 -.1 .2', '1 .5 .5 .1 .1',
    '0 nan .5 .1 .1', '0 inf .5 .1 .1', '0 .5 .5 .1',
    '0 0 0 1 1 2 2', '0 .1 .1 .2 .2 .3 .3',
    '0 -.01 .2 .5 .2 .3 .8', '0 1.1 .5 .1 .1',
])
def test_unresolved_rows_rejected(text):
    _,_,errors = parse_annotations(text)
    assert errors


def test_mixed_annotations_convert_without_losing_boxes():
    boxes,audit,errors = parse_annotations('0 .5 .5 .1 .1\n0 .1 .2 .5 .2 .3 .8')
    assert not errors and len(boxes)==2
    assert {row['format'] for row in audit}=={'bbox','polygon'}


def test_duplicate_objects_collapsed():
    boxes,_,errors = parse_annotations('0 .5 .5 .1 .1\n0 .5 .5 .1 .1')
    assert not errors and len(boxes)==1


def test_empty_has_no_objects():
    assert parse_annotations('\n  \n') == ([],[],[])


def test_source_normalization_and_scene_grouping():
    assert normalize_source('ABC_jpg.rf.123abc')=='abc'
    assert normalize_source('ship_0001')=='ship_0001'
    assert scene_key('0__20161102_180658_0e26__-122-328_37-739')=='acquisition:20161102_180658_0e26'
    assert scene_key('20190504_063211_1029_3B_Visual_0')=='visual:20190504_063211_1029_3B_Visual'
    assert scene_key('abc') is None


def test_bktree_matches_bruteforce_including_zero_distance():
    values = [0,1,2,3,7,15,255,1024,65535,0]
    tree = BKTree()
    for index,value in enumerate(values):
        tree.add(value,index)
    for query in [0,5,1023,65535]:
        for radius in [0,1,4]:
            assert sorted(tree.query(query,radius))==sorted((i,(query^v).bit_count()) for i,v in enumerate(values) if (query^v).bit_count()<=radius)


def test_transitive_union():
    uf = UnionFind(5)
    uf.union(0,1)
    uf.union(2,3)
    uf.union(1,3)
    assert len({uf.find(i) for i in range(4)})==1
    assert uf.find(4)!=uf.find(0)


def fixture_records():
    return [dict(id=f'i{i:03}',group_id=f'g{i//3:03}',source='aerial' if i%2 else 'ships',
                 width=768,height=768,boxes=[[0,.5,.5,.1,.1]]*(i%8+1)) for i in range(300)]


def test_grouped_split_reproducible_order_independent_and_balanced():
    records = fixture_records()
    a = grouped_split(records)
    assert a==grouped_split(records)==grouped_split(list(reversed(records)))
    assert split_digest(a)==split_digest(grouped_split(records))
    for group in {r['group_id'] for r in records}:
        assert len({a[r['id']] for r in records if r['group_id']==group})==1
    for split,target in [('train',.8),('val',.1),('test',.1)]:
        assert abs(sum(s==split for s in a.values())/len(records)-target)<.03


def test_ratio_validation():
    with pytest.raises(ValueError):
        grouped_split(fixture_records(),ratios=(.8,.1,.2))
