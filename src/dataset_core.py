"""Pure annotation, similarity, grouping and split primitives."""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy.fft import dctn


class AnnotationError(ValueError):
    pass


def normalize_source(stem: str) -> str:
    return re.sub(r'_(?:jpg|jpeg|png)\.rf\.[^.]+$', '', stem, flags=re.I).casefold()


def scene_key(stem: str) -> str | None:
    """Conservative parent acquisition key for geospatial chip filenames."""
    match = re.match(r'^\d+__(\d{8}_\d{6}_[^_]+)__', stem)
    if match:
        return 'acquisition:' + match[1]
    # Multiple chips from a large Visual scene have a shared timestamp/sensor.
    match = re.match(r'^(\d{8}_\d{6}_[^_]+_3B_Visual)(?:_|$)', stem)
    return 'visual:' + match[1] if match else None


def parse_annotations(text: str):
    """Return boxes and a full row audit; any unresolved row rejects the image.

    Polygon vertices outside the image are rejected, not silently clamped.
    Boxes are clipped in corner space; nonpositive input dimensions are rejected.
    """
    boxes, audit, errors = [], [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = {'line': number, 'original': line, 'format': 'unknown'}
        try:
            values = list(map(float, line.split()))
            if not values or not all(math.isfinite(x) for x in values):
                raise AnnotationError('non_finite_or_empty')
            if values[0] != 0:
                raise AnnotationError('invalid_class')
            if len(values) == 5:
                row['format'] = 'bbox'
                _, x, y, w, h = values
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise AnnotationError('zero_area_or_invalid_box')
                corners = [x-w/2, y-h/2, x+w/2, y+h/2]
            elif len(values) >= 7 and len(values) % 2 == 1:
                row['format'] = 'polygon'
                points = np.array(values[1:]).reshape(-1, 2)
                if not ((points >= 0).all() and (points <= 1).all()):
                    raise AnnotationError('polygon_out_of_bounds')
                # A closed polygon's repeated final point is harmless.
                area = abs(float(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
                                 np.dot(points[:, 1], np.roll(points[:, 0], 1)))) / 2
                if area <= 1e-12:
                    raise AnnotationError('zero_area_polygon')
                corners = [*points.min(axis=0), *points.max(axis=0)]
            else:
                raise AnnotationError('invalid_row_structure')
            x1, y1, x2, y2 = [min(1., max(0., v)) for v in corners]
            if x2-x1 <= 1e-9 or y2-y1 <= 1e-9:
                raise AnnotationError('zero_area_after_clipping')
            row['clipped'] = any(abs(a-b) > 1e-9 for a, b in zip(corners, [x1,y1,x2,y2]))
            box = [0, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
            row['box'] = box
            boxes.append(box)
        except (ValueError, OverflowError) as exc:
            row['error'] = str(exc)
            errors.append(f'line_{number}:{exc}')
        audit.append(row)
    # Exact duplicate object rows are safe to collapse after conversion.
    unique = {tuple(round(v, 10) for v in b): b for b in boxes}
    return list(unique.values()), audit, errors


def phash(image: Image.Image) -> int:
    gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(gray, type=2, norm='ortho')[:8, :8].ravel()
    bits = low > np.median(low[1:])
    bits[0] = False
    return int(''.join('1' if v else '0' for v in bits), 2)


class BKTree:
    """Exact Hamming-radius retrieval; avoids quadratic perceptual comparisons."""
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        result, stack = [], [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                result.extend((index, distance) for index in node[1])
            stack.extend(child for edge, child in node[2].items() if distance-radius <= edge <= distance+radius)
        return result


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a,b)] = min(a,b)


def annotation_signature(boxes):
    return sorted(tuple(round(v, 6) for v in row) for row in boxes)


def features(record):
    count = len(record['boxes'])
    w, h = record['width'], record['height']
    size = 'small' if max(w,h) <= 128 else 'medium' if max(w,h) <= 768 else 'large'
    density = 'one' if count == 1 else 'few' if count <= 5 else 'many'
    return ('images', 'source:'+record['source'], 'size:'+size, 'density:'+density,
            'aspect:'+('square' if .9 <= w/h <= 1.1 else 'wide' if w > h else 'tall'))


def grouped_split(records, seed=42, ratios=(.8,.1,.1)):
    """Deterministic greedy multivariate group allocation, then local refinement."""
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios)-1)>1e-9:
        raise ValueError('Three positive ratios summing to one are required')
    groups = defaultdict(list)
    for record in records:
        groups[record['group_id']].append(record)
    totals, vectors = Counter(), {}
    for key, rows in groups.items():
        vector = Counter(f for r in rows for f in features(r))
        vector['objects'] = sum(len(r['boxes']) for r in rows)
        vectors[key] = vector
        totals.update(vector)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))
    counts = [Counter() for _ in ratios]
    assignment = {}

    def cost(index, vector, sign=1):
        ratio = ratios[index]
        delta = 0.
        for f, n in vector.items():
            target = totals[f] * ratio
            before = counts[index][f] - target
            weight = 4. if f == 'images' else 1.
            delta += weight * ((before+sign*n)**2-before**2) / max(target,1.)
        return delta

    for key in keys:
        best = min(range(3), key=lambda i: (cost(i,vectors[key]), i))
        assignment[key] = best
        counts[best].update(vectors[key])
    for _ in range(8):
        changed = False
        for key in sorted(keys):
            old = assignment[key]
            best = min(range(3), key=lambda i: (0. if i == old else cost(old,vectors[key],-1)+cost(i,vectors[key]), i))
            if best != old and cost(old,vectors[key],-1)+cost(best,vectors[key]) < -1e-8:
                counts[old].subtract(vectors[key])
                counts[best].update(vectors[key])
                assignment[key] = best
                changed = True
        if not changed:
            break
    names = ['train','val','test']
    return {r['id']: names[assignment[r['group_id']]] for r in records}


def split_digest(assignment):
    payload = '\n'.join(f'{key}:{assignment[key]}' for key in sorted(assignment))
    return hashlib.sha256(payload.encode()).hexdigest()
