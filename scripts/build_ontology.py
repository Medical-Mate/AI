# -*- coding: utf-8 -*-
"""nodes.csv / edges.csv 재조립. obo 없이 돈다.

입력
- data/ontology/nodes.csv, edges.csv 의 UBERON:·MAN: 행 (extract_uberon.py 산출 + 수동 구조 11개)
- data/ontology/manual/nodes.csv, edges.csv (상부·앵커·표면 구역·located_in. 손으로 관리)

출력: 같은 자리의 nodes.csv / edges.csv 를 덮어쓴다. 컬럼 순서는 로더 규약과 같다.

규칙
- 구조 노드(UBERON·MAN)는 kind=structure, laterality=left_right 기본. 무릎·어깨 앵커는 kind=anchor
- REG:·ANC:·SUR: 는 manual/ 이 유일한 출처. 기존 파일에 있던 것은 버리고 manual/ 로 대체
- located_in 은 source 가 비어 있으면 로더가 조회에서 뺀다 (자문 회신 전 임시 채움 방지)
"""
import csv
import io
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'ontology')
MANUAL = os.path.join(OUT, 'manual')

NODE_COLS = ['id', 'name_en', 'name_ko', 'region', 'structure_type', 'sctid', 'fma', 'definition_en',
             'source', 'is_anchor', 'tier', 'kind', 'laterality']
EDGE_COLS = ['child', 'relation', 'parent', 'source']
ANCHORS_FROM_UBERON = {'UBERON:0001465', 'UBERON:0001467'}  # 무릎·어깨


def read(path):
    with io.open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, cols, rows):
    with io.open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def is_structure_row(r):
    return r['id'].startswith(('UBERON:', 'MAN:'))


nodes = []
for r in read(os.path.join(OUT, 'nodes.csv')):
    if not is_structure_row(r):
        continue
    if r['id'] in ANCHORS_FROM_UBERON:
        r.update(is_anchor='1', tier='2', kind='anchor', laterality='left_right')
    else:
        r.update(is_anchor='', tier='3', kind='structure', laterality=r.get('laterality') or 'left_right')
    nodes.append(r)
nodes += read(os.path.join(MANUAL, 'nodes.csv'))

edges = []
for r in read(os.path.join(OUT, 'edges.csv')):
    if r['child'].startswith(('UBERON:', 'MAN:')) and r['parent'].startswith(('UBERON:', 'MAN:')):
        edges.append({**r, 'source': r.get('source', '')})
manual_edges = read(os.path.join(MANUAL, 'edges.csv'))
seen = {(e['child'], e['relation'], e['parent']) for e in edges}
for e in manual_edges:
    key = (e['child'], e['relation'], e['parent'])
    if key in seen:
        continue
    seen.add(key)
    edges.append(e)

# 앵커 순서 = 인체도 첫 화면 배치(위→아래, 전신 마지막). 로더의 anchors_in_order()가 이 순서를 그대로 쓴다
ANCHOR_ORDER = ['ANC:001', 'ANC:002', 'ANC:003', 'ANC:004', 'ANC:005', 'UBERON:0001467', 'ANC:006',
                'ANC:007', 'ANC:008', 'UBERON:0001465', 'ANC:009', 'ANC:010']
KIND_ORDER = {'region': 0, 'anchor': 1, 'surface': 2, 'structure': 3}


def node_key(d):
    if d['kind'] == 'anchor':
        return (2, ANCHOR_ORDER.index(d['id']) if d['id'] in ANCHOR_ORDER else 99, d['id'])
    if d['kind'] == 'surface':
        return (3, KIND_ORDER['surface'], d['id'])  # SUR: 번호가 앵커 순서·구역 순서를 담는다
    return (int(d['tier']), KIND_ORDER[d['kind']], d['region'], d['id'])


nodes.sort(key=node_key)
edges.sort(key=lambda e: (e['relation'], e['child'], e['parent']))
write(os.path.join(OUT, 'nodes.csv'), NODE_COLS, nodes)
write(os.path.join(OUT, 'edges.csv'), EDGE_COLS, edges)

by_kind = {}
for n in nodes:
    by_kind[n['kind']] = by_kind.get(n['kind'], 0) + 1
print('노드 %d개 %s, 엣지 %d개' % (len(nodes), by_kind, len(edges)))
