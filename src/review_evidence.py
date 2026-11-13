"""Reproduce source-family semantic inspection panels from original imagery."""
import io
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def generate_review_evidence(records,config):
    groups = []
    for pattern in ['frederikshavn','skagen','3b_visual']:
        unique = {}
        for record in sorted(records,key=lambda r:r['id']):
            if record['source']=='ships' and pattern in record['normalized_source_name'] and record['boxes'] and not record.get('annotation_errors'):
                unique.setdefault(record['normalized_source_name'],record)
        pool = [unique[name] for name in sorted(unique)]
        if pool:
            second = pool[len(pool)//2]
            if pattern=='3b_visual':
                # Fixed reviewed example; show the suspect cloud-region labels
                # alongside a vessel example rather than implying all are wrong.
                second = unique.get('20190513_055612_1049_3b_visual_4',second)
            groups.append([pool[0],second])
    if not groups:
        return
    selected = [r for pair in groups for r in pair]
    fig,axes = plt.subplots(len(groups),2,figsize=(12,5*len(groups)),squeeze=False)
    with zipfile.ZipFile(config['sources']['ships']) as archive:
        for ax,record in zip(axes.flat,selected):
            with Image.open(io.BytesIO(archive.read(record['member']))) as image:
                image = image.convert('RGB')
                image.thumbnail((800,800))
                ax.imshow(image,extent=(0,1,1,0))
            for _,x,y,w,h in record['boxes']:
                ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fill=False,edgecolor='lime',linewidth=.8))
            ax.set_title(record['normalized_source_name']+'\nSource annotations for semantic review',fontsize=9)
            ax.set(xticks=[],yticks=[],xlim=(0,1),ylim=(1,0))
    fig.tight_layout()
    directory = config['reports']/'review'
    directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/'semantic_review.png',dpi=140,bbox_inches='tight')
    plt.close(fig)
    (directory/'semantic_review_images.json').write_text(json.dumps([
        {'id':r['id'],'source_name':r['normalized_source_name'],'member':r['member'],'status':r['status']}
        for r in selected],indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    from .prepare_dataset import ROOT,load_config,load_manifest
    config = load_config(ROOT/'configs/dataset.yaml')
    generate_review_evidence(load_manifest(config),config)
