# -*- coding: utf-8 -*-
"""온톨로지 매핑 현황 리포트. 팀 전달용 표를 마크다운으로 찍는다. LLM 호출 없음.

    uv run python scripts/ontology_report.py            # 콘솔
    uv run python scripts/ontology_report.py > out.md   # 파일

내용: 앵커 × 구역 표, 앵커별 구조 노드 수, located_in 채움률, 비어 있는 것(name_ko·sctid).
"""
import sys
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, 'src')
from medimate.ontology import load_ontology  # noqa: E402

o = load_ontology()
kinds = Counter(n.kind for n in o.nodes.values())
print('# 부위 온톨로지 현황 (snapshot %s)\n' % o.snapshot_id)
print('노드 %d = 상부 %d · 앵커 %d · 표면 구역 %d · 구조 %d. 검증 위반 %d건.\n' % (
    len(o), kinds['region'], kinds['anchor'], kinds['surface'], kinds['structure'], len(o.validate())))

print('## 인체도 입력: 앵커 → 구역 (환자가 누르는 것)\n')
print('| 면 | 상부 | 앵커 | 좌/우 | 구역 | 진료과 안내 (앵커 값) | 구조 노드 |')
print('|---|---|---|---|---|---|---|')
for a in o.anchors_in_order():
    region = o.nodes[o.region_of(a.id)].display_name
    no_zone = '(없음 — 사이드 탭)' if o.region_of(a.id) == 'REG:005' else '(없음 — 앵커만 누른다)'
    zones = ' / '.join(z.display_name for z in o.zones(a.id)) or no_zone
    n_struct = sum(1 for n in o.nodes.values() if n.kind == 'structure' and o.anchor_of(n.id) == a.id)
    depts = ' / '.join(a.departments) or '(구역 필수)'
    face = {'front': '앞', 'back': '뒤', 'none': '탭'}[a.view]
    print('| %s | %s | %s | %s | %s | %s | %d |' % (face, region, a.display_name, '○' if a.laterality == 'left_right' else '—', zones, depts, n_struct))

print('\n## 진료과 안내: 구역 값이 있는 것 (없으면 앵커 값을 쓴다)\n')
print('출처: 팀 결정 2026-09-04, 의료인 자문 확인 전. 우리 콘텐츠이고 인용이 아니다. 순서에 의미 없음.\n')
print('| 앵커 | 구역 | 진료과 안내 |')
print('|---|---|---|')
for a in o.anchors_in_order():
    for z in o.zones(a.id):
        if z.departments:
            print('| %s | %s | %s |' % (a.display_name, z.display_name, ' / '.join(z.departments)))
n_with = sum(1 for n in o.nodes.values() if n.departments)
n_zone_without = sum(1 for n in o.nodes.values() if n.kind == 'surface' and not n.departments)
print('\n- 값 있는 노드 %d개, 앵커 값을 쓰는 구역 %d개. 로더는 노출만 하고 고르지 않는다' % (n_with, n_zone_without))

print('\n## 구조 층: 차트 용어 → 앵커 (넓히기)\n')
print('구조 노드가 있는 앵커만. 다른 앵커는 UBERON 재추출 + 수동 보강이 필요하다.\n')
by_anchor = Counter(o.anchor_of(n.id) for n in o.nodes.values() if n.kind == 'structure')
for aid, cnt in sorted(by_anchor.items(), key=lambda kv: -kv[1]):
    print('- %s: %d개' % (o.nodes[aid].display_name, cnt))

print('\n## 두 층의 연결: located_in (구조 → 구역, 출처 필수)\n')
sourced = [e for e in o.located_in if e[2].strip()]
print('- 엣지 %d개, 출처 있는 것 %d개' % (len(o.located_in), len(sourced)))
print('- 채움은 의료인 자문·출처 확보 후. 출처 없는 엣지는 조회에서 자동 제외된다')

missing_ko = [n for n in o.nodes.values() if not n.name_ko]
missing_sctid = [n for n in o.nodes.values() if n.kind == 'structure' and not n.sctid]
print('\n## 비어 있는 것\n')
print('- name_ko 없음 %d개: %s' % (len(missing_ko), ', '.join(n.name_en for n in missing_ko)))
print('- 구조 노드 SCTID 없음 %d / %d' % (len(missing_sctid), kinds['structure']))
print('- 앵커·구역 SCTID/FMA: 전부 대조 전 (표면 국소해부 용어 확인 필요)')

print('\n## 대기 중인 결정 (이 데이터가 임시인 이유)\n')
print('- 앵커 9개(앞 6·뒤 1·탭 2)와 구역 경계는 디자이너 인체도 부위와 같아야 한다. 확정 시 manual/nodes.csv 수정')
print('- 위팔·아래팔·허벅지·종아리는 팔·다리 이미지가 전체라 생긴 칸. 빼면 팔 4·다리 3 구역')
print('- 구역별 증상 선택지는 온톨로지가 아니라 콘텐츠. 출처(국가건강정보포털) 라이선스 회신 후')
print('- 정신과·수면은 마네킹 밖. 자유 입력 + 자해 표현 감지 시 응급 안내 (자문 후 문구 확정)')
print('- 진료과 표는 의료인 자문 회신 시 전체 재검토 (docs/decisions/2026-09-04-department-guidance.md)')
