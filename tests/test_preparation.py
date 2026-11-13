from src.prepare_dataset import inspect_record, relate_and_deduplicate, serialize_boxes
from src.dataset_core import parse_annotations
from PIL import Image
import io


def record(index,text,source_name=None):
    image = Image.new('RGB',(32,32),'blue')
    stream = io.BytesIO()
    image.save(stream,format='PNG')
    value = {'id':str(index),'source':'aerial','original_split':'train_v2',
             'member':f'root/train_v2/{source_name or index}.png','raw_annotation':text}
    return inspect_record((value,stream.getvalue()))


def config():
    return dict(group_parent_scenes=True,phash_distance=4)


def test_missing_and_empty_never_become_backgrounds():
    assert 'missing_annotation' in record(0,None)['reasons']
    assert 'unverified_empty_annotation' in record(1,'\n')['reasons']


def test_one_bad_row_quarantines_entire_image():
    value = record(0,'0 .5 .5 .1 .1\n0 .5 .5 0 .1')
    relate_and_deduplicate([value],config())
    assert value['status']=='quarantined'


def test_conflicting_exact_duplicates_quarantined():
    values = [record(0,'0 .5 .5 .1 .1'),record(1,'0 .1 .1 .1 .1')]
    relate_and_deduplicate(values,config())
    assert all(v['status']=='quarantined' for v in values)
    assert values[0]['group_id']==values[1]['group_id']


def test_consistent_duplicate_retains_one_canonical():
    values = [record(0,'0 .5 .5 .1 .1'),record(1,'0 .5 .5 .1 .1')]
    relate_and_deduplicate(values,config())
    assert [v['status'] for v in values]==['accepted','duplicate_removed']
    assert values[1]['duplicate_of']=='0'


def test_invalid_exact_partner_cannot_silently_supply_label():
    values = [record(0,None),record(1,'0 .5 .5 .1 .1')]
    relate_and_deduplicate(values,config())
    assert all(v['status']=='quarantined' for v in values)


def test_serialization_survives_clipping_precision():
    boxes,_,_ = parse_annotations('0 .999999999 .5 .000001 .4')
    decoded,audit,errors = parse_annotations(serialize_boxes(boxes))
    assert decoded and not errors and not any(row['clipped'] for row in audit)


def test_serialization_at_decimal_rounding_ties():
    import numpy as np
    boxes,_,errors = parse_annotations('0 .27929687500000006 .51234549999999 .1 .1')
    restored,_,again = parse_annotations(serialize_boxes(boxes))
    assert not errors and not again
    assert np.allclose(boxes,restored,atol=1e-9,rtol=0)


def test_publish_retries_transient_windows_lock(tmp_path,monkeypatch):
    from pathlib import Path
    from src.prepare_dataset import publish_staging
    stage = tmp_path/'staging'
    output = tmp_path/'processed'
    stage.mkdir()
    (stage/'example.txt').write_text('preserved')
    real_rename = Path.rename
    attempts = []
    def rename(path,target):
        attempts.append(path)
        if len(attempts)==1:
            raise PermissionError('transient scanner lock')
        return real_rename(path,target)
    monkeypatch.setattr(Path,'rename',rename)
    monkeypatch.setattr('src.prepare_dataset.time.sleep',lambda _:None)
    publish_staging(stage,output)
    assert len(attempts)==2 and (output/'example.txt').read_text()=='preserved'
