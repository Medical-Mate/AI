# 부위 온톨로지 — 무릎 · 어깨

최초 추출 2026-09-01. 스크립트는 `../../scripts/extract_uberon.py`.

## 무엇인가

`docs/ai-design.md` §3의 부위 partonomy를 **UBERON에서 실제로 뽑아본 것**.
무릎·어깨 **40개 노드 / 39개 엣지**.

| 파일 | 내용 |
|---|---|
| `nodes.csv` | 노드. `id, name_en, name_ko, region, structure_type, sctid, fma, definition_en, source` |
| `edges.csv` | `child, relation, parent` — `part_of` / `is_a` |
| `../../scripts/extract_uberon.py` | 추출 스크립트. 저장소 루트에서 실행하되 `uberon-basic.obo`가 루트에 있어야 한다 |

## 원본

- **UBERON basic** — `http://purl.obolibrary.org/obo/uberon/basic.obo`
  12MB, 16,071 term, 2026-06-19 릴리스. `is_a` · `part_of` · `develops_from`만 남긴 서브셋
- 라이선스 확인 필요 (OBO Foundry 계열, CC-BY 추정 — **단정하지 말 것**)

## 확인된 것

- **partonomy가 실제로 돈다.** ACL → 십자인대 → 무릎관절 인대 →(part_of)→ 무릎관절 →(part_of)→ 무릎 →(part_of)→ 다리
- **SNOMED CT ID(SCTID)가 xref로 붙어 있다** — 40개 중 23개.
  **SNOMED CT 라이선스 없이도 ID 매핑 경로가 확보된다.** FMA·UMLS·MeSH·NCIT도 같이 붙는다
- 규모가 설계와 맞는다 — 무릎 19 / 어깨 21. `docs/ai-design.md`의 "30~50개"와 일치

## ★ 중요 — **UBERON만으로는 임상적으로 불완전하다**

UBERON은 **다종(multi-species) 비교해부학** 온톨로지다. 사람 임상에서 흔한 구조가 빠진다.
`source` 컬럼이 `수동(...)`인 **11개**가 그것이다.

| 빠진 것 | 사유 |
|---|---|
| **내측·외측 측부인대(MCL/LCL)** | **UBERON에 아예 없다.** 무릎 통증에서 ACL만큼 흔한데 |
| **반월판(내측·외측)** | `meniscus`는 있으나 `part_of skeletal joint` — **무릎 하위가 아니다** |
| 무릎 관절연골 | 없음 |
| 견봉하 점액낭, 위팔두갈래근 긴갈래 힘줄 | 없음 |
| 어깨뼈·빗장뼈·위팔뼈·봉우리 | UBERON에선 `pectoral girdle` 하위라 **어깨 subtree 밖** |

**이게 나쁜 소식이 아니다.** 30~50개 규모면 손보강이 충분히 가능하고,
**"기존 온톨로지가 임상적으로 불완전하다"는 사실이 우리가 만드는 것의 정당성**이 된다.
보강분이 곧 고유 자산이다.

## 다음에 할 것

1. **`name_ko` 검증** — 40개 중 36개를 채웠으나 **대한해부학회 「해부학용어」 대조 전이다.**
   한자어/순우리말 두 계열이 병기돼 있는데(슬개골/무릎뼈), **환자는 대개 순우리말이나 일상어를 쓴다**
2. **수동 11개의 SCTID·FMA 채우기** — SNOMED CT 브라우저 또는 FMA에서
3. **환자 표현은 앵커 노드에만 붙인다** — 무릎·어깨 2개.
   세부 38개는 "차트에서 인식"만 하면 되므로 이름·타입·SNOMED ID로 충분하다
   ( 「번역 방향」 — 좁히지 않고 넓힌다)
4. ** 컬럼 추가** — 어디서 멈출지. 디자이너 인체도 부위와 같아야 한다
4. **`common-confusion` 엣지** — ACL ↔ 반월판 ↔ 슬개골. 자문 회신 후
5. **`laterality`** — 좌/우. 지금 노드에 없다. 속성으로 넣을지 노드로 쪼갤지 결정
6. **`is_anchor` 컬럼 추가** — 올라가다 어디서 멈출지. **디자이너 인체도 부위와 같아야 한다**
7. **디자이너 인체도가 나오면 그 부위 단위에 맞춰 조정** (`docs/ai-design.md` §7)

## 안 한 것

- 발생학(primordium·mesenchyme), 조직 수준(epithelium), 동물(stifle joint) 제외
- **혈관·신경·림프절 제외** — 환자가 그 이름으로 말하지 않는다
- `develops_from` 관계 미사용 — 발생학이라 불필요
