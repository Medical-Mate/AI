import re, csv, collections, io, os

# 저장소 루트 기준. uberon-basic.obo 도 루트에 두고 실행한다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "ontology")
OBO = os.path.join(ROOT, "uberon-basic.obo")
os.makedirs(OUT, exist_ok=True)

def clean(v):
    return re.sub(r'\s*[!{].*$', '', v).strip()

T = {}
cur = None
for L in open(OBO, encoding='utf-8'):
    L = L.rstrip('\n')
    if L == '[Term]':
        cur = {'id': None, 'name': None, 'is_a': [], 'part_of': [], 'xref': [], 'def': ''}
    elif L.startswith('['):
        cur = None
    elif cur is not None and ': ' in L:
        k, _, v = L.partition(': ')
        if k == 'id':
            cur['id'] = clean(v); T[cur['id']] = cur
        elif k == 'name':
            cur['name'] = v
        elif k == 'is_a':
            cur['is_a'].append(clean(v))
        elif k == 'relationship' and v.startswith('part_of '):
            cur['part_of'].append(clean(v.split(None, 1)[1]))
        elif k == 'xref':
            cur['xref'].append(v.split(' ')[0])
        elif k == 'def':
            cur['def'] = v.split('" [')[0].strip('"')

nm = lambda i: T.get(i, {}).get('name', '?')

kids = collections.defaultdict(list)
for t in T.values():
    for p in t['part_of'] + t['is_a']:
        if p in T:
            kids[p].append(t['id'])

def sub(r):
    s = set(); st = [r]
    while st:
        x = st.pop()
        if x in s: continue
        s.add(x); st += kids.get(x, [])
    return s

DROP = re.compile(
    r'primordium|pre-cartilage|pre-muscle|mesenchyme|epithelium|'
    r'skeletal muscle tissue|stifle|cartilage element|obsolete|venous plexus|'
    r'lymph node|metacromion|deltoid process|artery|vein|nerve|interscapular fat pad', re.I)

KO = {
    'knee': '무릎', 'knee joint': '무릎관절(슬관절)', 'patella': '무릎뼈(슬개골)',
    'anterior cruciate ligament of knee joint': '앞십자인대(전방십자인대)',
    'posterior cruciate ligament of knee joint': '뒤십자인대(후방십자인대)',
    'cruciate ligament of knee': '십자인대', 'ligament of knee joint': '무릎관절 인대',
    'patellar ligament': '무릎인대(슬개인대)',
    'femorotibial joint': '넙다리정강관절(대퇴경골관절)',
    'patellofemoral joint': '무릎넙다리관절(슬개대퇴관절)',
    'infrapatellar fat pad': '무릎아래지방체(슬개하지방체)',
    'skin of knee': '무릎 피부', 'knee connective tissue': '무릎 결합조직',
    'meniscus': '반달연골(반월판)',
    'shoulder': '어깨', 'shoulder joint': '어깨관절(견관절)',
    'glenohumeral joint': '오목위팔관절(관절와상완관절)',
    'scapula': '어깨뼈(견갑골)', 'clavicle': '빗장뼈(쇄골)', 'humerus': '위팔뼈(상완골)',
    'acromion': '봉우리(견봉)', 'rotator cuff': '돌림근띠(회전근개)',
    'supraspinatus muscle': '가시위근(극상근)', 'infraspinatus muscle': '가시아래근(극하근)',
    'subscapularis muscle': '어깨밑근(견갑하근)', 'teres minor muscle': '작은원근(소원근)',
    'glenoid labrum of scapula': '오목테두리(관절와순)',
    'coracoclavicular ligament': '부리빗장인대(오구쇄골인대)',
    'infraspinatus tendon': '가시아래근힘줄', 'skin of shoulder': '어깨 피부',
}

def stype(n):
    n = n.lower()
    pairs = [('ligament', '인대'), ('tendon', '힘줄'), ('cuff', '근육군'), ('muscle', '근육'),
             ('labrum', '연골'), ('meniscus', '연골'), ('cartilage', '연골'), ('bursa', '활액낭'),
             ('capsule', '관절주머니'), ('fat pad', '지방체'), ('skin', '피부'),
             ('connective', '결합조직'), ('joint', '관절')]
    for k, v in pairs:
        if k in n: return v
    if n in ('patella', 'scapula', 'clavicle', 'humerus', 'acromion'): return '뼈'
    if n in ('knee', 'shoulder'): return '부위'
    return ''

nodes = {}
edges = []
for root, region in [('UBERON:0001465', '무릎'), ('UBERON:0001467', '어깨')]:
    for x in sub(root):
        n = nm(x)
        if DROP.search(n): continue
        nodes[x] = {
            'id': x, 'name_en': n, 'name_ko': KO.get(n, ''), 'region': region,
            'structure_type': stype(n),
            'sctid': next((r.split(':')[1] for r in T[x]['xref'] if r.startswith('SCTID')), ''),
            'fma': next((r.split(':')[1] for r in T[x]['xref'] if r.startswith('FMA')), ''),
            'definition_en': T[x]['def'][:200], 'source': 'UBERON',
            # 앵커(환자 어휘 수준, docs/ai-design.md §3). 디자이너 인체도 확정 전 임시로 무릎·어깨만
            'is_anchor': '1' if x in ('UBERON:0001465', 'UBERON:0001467') else '',
            # 등급: 1 상부(다부위 정리) / 2 앵커(환자 어휘) / 3 세부(차트 인식). 그래프 깊이가 아니다
            'tier': '2' if x in ('UBERON:0001465', 'UBERON:0001467') else '3',
        }

for x in list(nodes):
    for p in T[x]['part_of']:
        if p in nodes: edges.append((x, 'part_of', p))
    for p in T[x]['is_a']:
        if p in nodes: edges.append((x, 'is_a', p))

MAN = [
    ('MAN:001', 'medial collateral ligament of knee', '안쪽곁인대(내측측부인대)', '무릎', '인대', 'UBERON에 없음 - FMA/SNOMED 확인 필요'),
    ('MAN:002', 'lateral collateral ligament of knee', '가쪽곁인대(외측측부인대)', '무릎', '인대', 'UBERON에 없음'),
    ('MAN:003', 'medial meniscus', '안쪽반달연골(내측반월판)', '무릎', '연골', 'UBERON meniscus는 무릎 하위가 아님'),
    ('MAN:004', 'lateral meniscus', '가쪽반달연골(외측반월판)', '무릎', '연골', '같음'),
    ('MAN:005', 'articular cartilage of knee', '무릎 관절연골', '무릎', '연골', 'UBERON에 없음'),
    ('MAN:006', 'subacromial bursa', '봉우리밑주머니(견봉하 점액낭)', '어깨', '활액낭', 'UBERON에 없음'),
    ('MAN:007', 'biceps long head tendon', '위팔두갈래근 긴갈래 힘줄', '어깨', '힘줄', '어깨 하위 미포함'),
    ('MAN:008', 'scapula', '어깨뼈(견갑골)', '어깨', '뼈', 'UBERON은 pectoral girdle 하위 - 어깨 subtree 밖'),
    ('MAN:009', 'clavicle', '빗장뼈(쇄골)', '어깨', '뼈', '같음'),
    ('MAN:010', 'humerus', '위팔뼈(상완골)', '어깨', '뼈', '같음'),
    ('MAN:011', 'acromion', '봉우리(견봉)', '어깨', '뼈', '같음'),
]
for i, en, ko, reg, st, note in MAN:
    nodes[i] = {'id': i, 'name_en': en, 'name_ko': ko, 'region': reg, 'structure_type': st,
                'sctid': '', 'fma': '', 'definition_en': '', 'source': '수동(' + note + ')',
                'is_anchor': '', 'tier': '3'}

# 상부·앵커·표면 구역은 data/ontology/manual/ 이 출처다. 추출 후 scripts/build_ontology.py 로 합친다

edges += [('MAN:001', 'part_of', 'UBERON:0001485'), ('MAN:002', 'part_of', 'UBERON:0001485'),
          ('MAN:003', 'part_of', 'UBERON:0001485'), ('MAN:004', 'part_of', 'UBERON:0001485'),
          ('MAN:005', 'part_of', 'UBERON:0001485'), ('MAN:006', 'part_of', 'UBERON:0001467'),
          ('MAN:007', 'part_of', 'UBERON:0001467'),
          ('MAN:008', 'part_of', 'UBERON:0001467'), ('MAN:009', 'part_of', 'UBERON:0001467'),
          ('MAN:010', 'part_of', 'UBERON:0001467'), ('MAN:011', 'part_of', 'MAN:008')]

cols = ['id', 'name_en', 'name_ko', 'region', 'structure_type', 'sctid', 'fma', 'definition_en', 'source',
        'is_anchor', 'tier']
with io.open(os.path.join(OUT, 'nodes.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for x in sorted(nodes.values(), key=lambda d: (d['region'], d['name_en'])):
        w.writerow(x)

with io.open(os.path.join(OUT, 'edges.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['child', 'relation', 'parent'])
    for e in sorted(set(edges)):
        w.writerow(e)

print('노드 %d개 (UBERON %d + 수동 %d), 엣지 %d개'
      % (len(nodes), len(nodes) - len(MAN), len(MAN), len(set(edges))))
print('한국어 채움 %d/%d, SNOMED ID 있음 %d'
      % (sum(1 for n in nodes.values() if n['name_ko']), len(nodes),
         sum(1 for n in nodes.values() if n['sctid'])))
print()
for reg in ['무릎', '어깨']:
    rows = sorted([n for n in nodes.values() if n['region'] == reg], key=lambda d: d['name_en'])
    print('=== %s (%d) ===' % (reg, len(rows)))
    for n in rows:
        flag = '※' if n['source'].startswith('수동') else ' '
        print(' %s %-26s %-42s %-6s %s'
              % (flag, n['name_ko'] or '(미기입)', n['name_en'][:40], n['structure_type'], n['sctid']))
    print()
