"""부위 검색 테스트 벡터 생성 — 앱이 로컬 포팅을 검증할 기준.

    uv run python scripts/gen_body_search_vectors.py

한/영 자판 오타 복원은 넣지 않는다. 서버만 하는 기능이고 앱은 포팅하지 않기로 했다(2026-09-11).
결과는 docs/examples/body-search-vectors.json. 커밋된 파일과 현재 동작이 같은지는 테스트가 지킨다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medimate.ontology.graph import load_ontology  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "examples" / "body-search-vectors.json"
LIMIT = 8

EXTRA = [
    "왼쪽 아랫배가 아파요",  # 문장 안에서 구체적인 구역이 먼저
    "아랫배가 아파요",
    "무릎이 시큰거려요",
    "전체",  # 여러 앵커에 약하게 걸린다 — 구역을 펼치지 않아야 한다
    "허리",
    "목",
    "다리",
    "인대",  # 구조 노드는 검색 대상이 아니다
    "무릅",  # 한글 오타는 잡지 않는다
    "감기",  # 증상·병명은 매칭하지 않는다
    "zzzz",
    "",
    "   ",
]


def build() -> dict:
    onto = load_ontology()
    terms: list[str] = []
    for node in onto.nodes.values():
        if node.kind not in ("anchor", "surface"):
            continue
        terms.append(node.name_ko)
        terms.extend(node.aliases)
    seen: set[str] = set()
    queries = [t for t in terms + EXTRA if not (t in seen or seen.add(t))]

    cases = []
    for q in queries:
        hits = onto.search(q, limit=LIMIT)
        cases.append(
            {
                "q": q,
                "expect": [{"id": n.id, "matched": m, "score": s} for n, m, s in hits],
            }
        )
    return {
        "schema": "medimate-body-search-vectors/1",
        "ontology_snapshot": onto.snapshot_id,
        "limit": LIMIT,
        "note": (
            "부위 폼 검색의 기대 결과. 순서까지 일치해야 한다. "
            "앱이 body-map의 label+aliases로 로컬 매칭을 구현할 때 이 벡터로 검증한다. "
            "한/영 자판 오타 복원(qo→배)은 서버 전용이라 벡터에 없다."
        ),
        "cases": cases,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(Path.cwd())} — {len(build()['cases'])} cases")
