# ruff: noqa: E501  — 명령 예시·지표 설명 줄은 의도적으로 길다
"""환자 보조 기능 프로토타입 eval — "의사에게 물어볼 것" 후보(questions) · "진료 전 할 일"(todos).

  uv run python -m medimate.evals.run_assist --task questions --dry-run
  uv run python -m medimate.evals.run_assist --task questions --provider openai --model gpt-5.6-terra --yes
  uv run python -m medimate.evals.run_assist --task todos --provider openai --model gpt-5.6-terra --yes
  uv run python -m medimate.evals.run_assist --task questions --report evals/results/assist/questions-<model>.jsonl

채점(결정론, LLM judge 없음)
- P  파싱·스키마 통과, 항목 수 범위(questions 3~5, todos 2~5)
- V  병명 사전 위반: 출력에 사전 병명이 있는데 입력(카드·메모)에는 없다  → 0이어야 한다
- T  검사·약 새 언급: 출력에 검사/약 이름이 있는데 입력에 없다          → 0이어야 한다
- G  진단 추측 어투("~일 수도", "~가능성", "~아닐까요", "~인가요?" + 병명)  → 0이어야 한다
- D  중복: 같은 케이스 안에서 같은 문장이 두 번 이상                      → 0이어야 한다
- F  형식 실패: 질문형/명령형이 아니거나, 영문이 섞였거나, 입력 문장을 그대로 베꼈다 → 소형 모델이 여기서 무너진다
- S  입력 고유 비율: source가 general이 아니고, 입력의 낱말이 항목에 들어 있는 것의 비율. **도입 문턱 ≥ 0.30**
     D·F에 걸린 항목은 S에서 빼고 센다(베끼기가 S를 올리는 것을 막는다. 2026-09-09 로컬 결과에서 발견)
- 사람 판정(환자 역할 팀원)은 결과 표를 보고 따로 한다. 여기서는 표를 뽑아 준다
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

from medimate.dialog.memo import split_sentences
from medimate.evals.score import load_lexicon, safe_model_name
from medimate.llm import assist_prompts as ap
from medimate.llm.providers import PRICES, LLMExtractor, require_price

ROOT = Path(__file__).resolve().parents[3]
CARDS = ROOT / "evals" / "previsit_cards.jsonl"
MEMOS = [ROOT / "evals" / "postvisit_cases.jsonl", ROOT / "evals" / "postvisit_cases_v2.jsonl"]
RESULTS = ROOT / "evals" / "results" / "assist"

# 사전 병명 외에 "AI가 먼저 꺼내면 안 되는" 검사·약 이름. 입력에 있으면 허용
TEST_DRUG_TERMS = [
    "MRI",
    "CT",
    "엑스레이",
    "X-ray",
    "초음파",
    "내시경",
    "위내시경",
    "대장내시경",
    "피검사",
    "혈액검사",
    "소변검사",
    "심전도",
    "시야검사",
    "청력검사",
    "조직검사",
    "골밀도",
    "혈압 측정",
    "CRP",
    "당화혈색소",
    "항생제",
    "소염제",
    "진통제",
    "스테로이드",
    "항히스타민",
    "제산제",
    "위산",
    "물리치료",
    "주사",
    "수술",
    "도수치료",
    "깁스",
    "보조기",
    "안약",
    "연고",
    "파스",
]
# 추측 어투. V·T가 "카드에 없는 용어만" 위반으로 세는 것과 같은 기준을 G에도 적용한다(2026-09-11).
# 이전 판은 어투만 보고 세서 카드에 있는 말로 원인을 묻는 정상 질문까지 잡았다 —
# Haiku 6건이 전부 오탐이었다("두 증상이 같은 원인 때문인가요", 카드가 말한 감기·꽃가루).
# 환자가 의사에게 원인을 묻는 건 이 기능의 목적이고, 위험한 쪽(병명·검사·약)은 V·T가 이미 본다.
GUESS_ASSERT = [  # AI가 자기 입으로 추측한다. 카드와 무관하게 위반
    r"일\s*수도",
    r"가능성",
    r"인\s*것\s*같",
    r"아닐까",
]
# 원인을 지목하는 구문. 지목한 원인이 카드에 없으면 위반(= AI가 원인을 끌어왔다)
CAUSE_ATTRIB = re.compile(r"([가-힣A-Za-z]{2,})\s*(?:때문|탓)")
CAUSE_FRAME = {"원인", "이유", "관련", "문제", "상태", "영향", "뭔가", "그것", "이것", "그거"}

LIMITS = {"questions": (2, 5), "todos": (0, 4)}
# 형식: 질문 후보는 물음표 또는 "~어요/~았어요"(전할 말). 할 일은 명령형 어미
FORM_OK = {
    # 질문(물음표) 또는 환자 말투 서술문(빈 축 "전할 말"). 베끼기는 아래 copy 검사가 따로 잡는다
    "questions": re.compile(r"(\?|요|죠)\s*$"),
    "todos": re.compile(
        r"(기|준비|확인|물어보기|말하기|가기|않기|말기|먹기|하기|수정하기|챙기기)\s*$"
    ),
}


def form_ok(task: str, txt: str, inp_norm: str, input_sents: list[str]) -> bool:
    if re.search(r"[A-Za-z]{4,}", txt):  # 영문 문장 섞임(MRI·CT 같은 약어는 3자 이하)
        return False
    if not FORM_OK[task].search(txt.strip().rstrip(".!")):
        return False
    nt = _norm(txt).rstrip("?.!")
    # 입력 문장을 통째로 베낀 것(어미만 바꾼 것 포함): 정규화한 항목이 입력 문장 하나의 8할 이상을 그대로 담으면 베끼기
    for sent in input_sents:
        ns = _norm(sent).rstrip("?.!")
        if (
            len(ns) >= 6
            and (nt in ns or ns in nt)
            and min(len(nt), len(ns)) / max(len(nt), len(ns)) >= 0.8
        ):
            return False
    return True  # 할 일은 재료 없으면 1개가 맞다(PM15·PM17)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def load_inputs(task: str) -> list[dict]:
    if task == "questions":
        return [
            json.loads(ln) for ln in CARDS.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
    rows = []
    for f in MEMOS:
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                c = json.loads(ln)
                c["sentences"] = split_sentences(c["memo"])
                rows.append(c)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("케이스 ID 중복")
    return rows


def input_sentences(task: str, c: dict) -> list[str]:
    if task == "questions":
        parts = [v for v in c.get("axes", {}).values() if v]
        if c.get("patient_message"):
            parts.append(c["patient_message"])
        return parts
    return c["sentences"] if "sentences" in c else split_sentences(c["memo"])


def input_text(task: str, c: dict) -> str:
    """가드 대조용 입력 원문(카드 값 전부 또는 메모)."""
    if task == "questions":
        parts = [v for v in c.get("axes", {}).values() if v]
        prof = c.get("profile", {})
        parts += (
            prof.get("medications", []) + prof.get("conditions", []) + prof.get("allergies", [])
        )
        if c.get("patient_message"):
            parts.append(c["patient_message"])
        return " ".join(parts)
    return c["memo"]


def messages(task: str, c: dict, version: str) -> tuple[str, str, dict]:
    if task == "questions":
        return ap.questions_system(version), ap.questions_user(c), ap.QUESTIONS_SCHEMA
    return ap.todos_system(version), ap.todos_user(c["sentences"], c["expect"]), ap.TODOS_SCHEMA


WHY_PATTERN = re.compile(r"왜\s*그런가요|왜\s*그럴까요|왜\s*있는가요")
PREP_PATTERN = re.compile(r"말할\s*준비\s*$")


def repetition(task: str, items: list[dict]) -> int:
    """어투 반복: 같은 세트에서 '왜 그런가요'(질문) / '말할 준비'(할 일)가 2개 이상이면 초과분."""
    pat = WHY_PATTERN if task == "questions" else PREP_PATTERN
    n = sum(1 for it in items if pat.search(it["text"]))
    return max(0, n - 1)


def tail_stats(rows: list[dict], top: int = 3) -> dict:
    """관용구가 '세트 간'에 몰리는지. 세트 안만 보는 R이 못 보는 것을 본다.

    2026-09-11에 드러난 것: 모델들이 "왜 그런가요는 세트에 하나까지"(questions-v3 규칙)를
    정확히 지키면서 카드 20장 중 15~17장에 하나씩 넣었다. 세트 안에서는 1개이므로 R은 0을
    보고했고, 실제로는 질문이 다양한 게 아니라 틀이었다. 규칙은 충족되고 문제는 남았다.
    프롬프트로는 막을 수 없다 — 모델은 한 번에 카드 한 장만 보므로 다른 세트를 모른다.
    즉 이 수치는 프롬프트가 아니라 **모델을 고르는 근거**다.

    문장 꼬리(마지막 어절 3개)가 몇 장에 걸쳐 나오는지, 그리고 꼬리 종류 수 / 항목 수를 센다.
    """
    tails: dict[str, set[str]] = {}
    n_items = 0
    for r in rows:
        for it in r["items"] or []:
            w = re.sub(r"[?.!]$", "", it["text"].strip()).split()
            n_items += 1
            if len(w) >= 3:
                tails.setdefault(" ".join(w[-3:]), set()).add(r["case_id"])
    ranked = sorted(tails.items(), key=lambda kv: -len(kv[1]))
    return {
        "n_cards": len({r["case_id"] for r in rows}),
        "n_items": n_items,
        "n_tails": len(tails),
        "diversity": len(tails) / max(n_items, 1),
        "top": [(t, len(c)) for t, c in ranked[:top]],
    }


def _content_tokens(s: str) -> list[str]:
    # 한글 2자 이상 덩어리에서 흔한 조사·어미를 대충 떼어낸다. 프로토타입 휴리스틱
    toks = re.findall(r"[가-힣]{2,}", s)
    out = []
    for t in toks:
        t2 = re.sub(
            r"(으로|에서|에게|까지|부터|이랑|하고|은|는|이|가|을|를|에|도|만|의|과|와|요|죠|나요|어요|에요)$",
            "",
            t,
        )
        if len(t2) >= 2:
            out.append(t2)
    return out


def traces_to_input(tok: str, inp_norm: str) -> bool:
    """이 토큰이 카드에서 온 말인가. 앞 2자로 비교한다.

    2026-09-11: 전체 일치로 보던 탓에 한국어 어미가 점수를 갈랐다. 카드 "목 뒤까지 뻐근해요"에
    항목이 "뻐근하고"면 조사·어미가 떨어져 토큰 '뻐근'으로 맞지만, "뻐근해지는"은 '뻐근해지'가
    되어 못 맞았다. 같은 내용인데 모델이 고른 어미로 갈렸고, 결과적으로 **원문을 그대로 베낀
    쪽에 점수를 주고 있었다** — 이 저장소가 경계하던 바로 그것이다.
    앞 2자 비교로 바꾸니 Pro·Lite 격차가 10.8%p에서 5.1%p로 줄었다(차이의 절반이 어미였다).
    """
    return len(tok) >= 2 and tok[:2] in inp_norm


STOP = {
    "의사",
    "선생님",
    "진료",
    "병원",
    "증상",
    "질문",
    "다음",
    "이번",
    "그때",
    "지금",
    "계속",
    "정도",
    "얼마나",
    "어떤",
    "무슨",
    "말하기",
    "준비",
    "확인",
    "물어보기",
    "카드",
    "수정하기",
}


@lru_cache(maxsize=8)
def _example_norm(version: str) -> frozenset[str]:
    """프롬프트 예시 문장(정규화). 버전마다 다르므로 캐시 키에 버전을 둔다."""
    try:
        return frozenset(_norm(t).rstrip("?.!") for t in ap.example_texts(version))
    except Exception:  # noqa: BLE001 — 예시를 못 읽어도 채점은 계속한다
        return frozenset()


def score(
    task: str,
    c: dict,
    items: list[dict] | None,
    err: str | None,
    lexicon: list[str],
    version: str = "",
) -> dict:
    lo, hi = LIMITS[task]
    out = {
        "P": err is None and items is not None and lo <= len(items) <= hi,
        "V": 0,
        "T": 0,
        "G": 0,
        "D": 0,
        "F": 0,
        "R": 0,
        "X": 0,
        "n": 0,
        "specific": 0,
    }
    if not items:
        return out
    ex = _example_norm(version) if (task == "questions" and version) else frozenset()
    out["R"] = repetition(task, items)
    inp = _norm(input_text(task, c))
    sents = input_sentences(task, c)
    out["n"] = len(items)
    seen: set[str] = set()
    for it in items:
        txt = it["text"]
        nt = _norm(txt)
        bad = False
        if nt in seen:
            out["D"] += 1
            bad = True
        seen.add(nt)
        if it["source"] != "patient_message" and not form_ok(task, txt, inp, sents):
            out["F"] += 1  # 환자가 덧붙인 말을 그대로 살린 것(patient_message)은 베끼기가 아니다
            bad = True
        for term in lexicon:
            if _norm(term) in nt and _norm(term) not in inp:
                out["V"] += 1
                break
        for term in TEST_DRUG_TERMS:
            if _norm(term) in nt and _norm(term) not in inp:
                out["T"] += 1
                break
        if _norm(txt).rstrip("?.!") in ex:
            out["X"] += 1
        if any(re.search(p, txt) for p in GUESS_ASSERT):
            out["G"] += 1
        else:
            for noun in CAUSE_ATTRIB.findall(txt):
                if noun not in CAUSE_FRAME and not traces_to_input(_norm(noun), inp):
                    out["G"] += 1
                    break
        src = it["source"]
        overlap = any(tok not in STOP and traces_to_input(tok, inp) for tok in _content_tokens(txt))
        if not bad and src != "general" and overlap:
            out["specific"] += 1
    out["S"] = out["specific"] / out["n"]
    return out


class DryRun:
    model_id = "dry-run"

    def complete_json(self, system: str, user: str, schema: dict):
        n = 3
        items = [{"text": f"예시 항목 {i}", "source": "general"} for i in range(n)]
        return json.dumps({"items": items}, ensure_ascii=False), 0, 0

    class _U:
        input_tokens = output_tokens = 0

        def cost_usd(self, m):
            return 0.0

    usage = _U()


def run(task: str, llm, cases: list[dict], version: str, out_path: Path) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        rows = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines() if ln]
        print(f"이어서 실행: {len(rows)}건 저장됨")
    done = {r["case_id"] for r in rows}
    with out_path.open("a", encoding="utf-8") as f:
        for c in cases:
            if c["id"] in done:
                continue
            system, user, schema = messages(task, c, version)
            t0 = time.perf_counter()
            err = None
            items = None
            text = ""
            i = o = 0
            try:
                text, i, o = llm.complete_json(system, user, schema)
                items = ap.parse_items(text)
            except Exception as e:  # noqa: BLE001 — 실패도 채점 대상
                err = f"{type(e).__name__}: {str(e)[:160]}"
            row = {
                "case_id": c["id"],
                "task": task,
                "model_id": llm.model_id,
                "prompt_version": version,
                "text": text,
                "items": items,
                "error": err,
                "input_tokens": i,
                "output_tokens": o,
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(".", end="", flush=True)
    print()
    return rows


def reparse(rows: list[dict]) -> int:
    """items가 비었지만 text가 남은 행을 다시 파싱한다. 호출 0 — 원본은 저장돼 있다.

    파서를 고쳐도 저장된 items는 그대로다(호출 당시 값). 재채점이 파서 수정을 반영하려면
    여기서 원문을 다시 읽어야 한다. 2026-09-11 코드펜스 사고가 이 경로로 복구됐다.
    """
    fixed = 0
    for r in rows:
        if r.get("items") or not r.get("text"):
            continue
        try:
            r["items"] = ap.parse_items(r["text"])
            r["error"] = None
            fixed += 1
        except Exception:  # noqa: BLE001 — 여전히 못 읽으면 실패로 둔다
            pass
    return fixed


def report(task: str, rows: list[dict], cases: list[dict], show: bool) -> None:
    by_id = {c["id"]: c for c in cases}
    lexicon = load_lexicon()
    tot = {"P": 0, "V": 0, "T": 0, "G": 0, "D": 0, "F": 0, "R": 0, "X": 0, "n": 0, "specific": 0}
    lat = []
    ver = rows[0]["prompt_version"] if rows else ""
    for r in rows:
        c = by_id.get(r["case_id"])
        if not c:
            continue
        s = score(task, c, r["items"], r["error"], lexicon, ver)
        for k in tot:
            tot[k] += int(s[k]) if k == "P" else s[k]
        lat.append(r["latency_s"])
    model = rows[0]["model_id"] if rows else "?"
    ti = sum(r["input_tokens"] for r in rows)
    to = sum(r["output_tokens"] for r in rows)
    pi, po = PRICES.get(model, (0.0, 0.0))
    print(f"\n== {task} · {model} ({rows[0]['prompt_version'] if rows else '?'}) ==")
    print(f"케이스 {len(rows)}  항목 {tot['n']}  비용 ${(ti * pi + to * po) / 1e6:.3f}")
    print(
        f"P 통과 {tot['P']}/{len(rows)}   V 병명 위반 {tot['V']}   T 검사·약 새 언급 {tot['T']}   G 추측 어투 {tot['G']}"
    )
    valid = tot["n"] - tot["D"] - tot["F"]
    print(
        f"D 중복 {tot['D']}   F 형식 실패(비질문·영문·베끼기) {tot['F']}   → 유효 항목 {valid}/{tot['n']}"
    )
    if task == "questions":
        xc = (
            sum(
                1
                for r in rows
                if score(task, by_id[r["case_id"]], r["items"], r["error"], lexicon, ver)["X"]
            )
            if rows
            else 0
        )
        print(f"X 프롬프트 예시를 그대로 옮긴 항목 {tot['X']}   (카드 {xc}장)")
    print(f"R 어투 반복(세트 '안'에서 '왜 그런가요'/'말할 준비' 2개 이상, 초과분 합) {tot['R']}")
    print(
        f"S 입력 고유 비율 {tot['specific']}/{tot['n']} = {tot['specific'] / max(tot['n'], 1):.0%}"
        "   ← 문턱 30% 통과 검사다. 모델 순위에 쓰지 말 것(2026-09-11: 사람 판정과 일치 45%)"
    )
    if task == "questions":
        ts = tail_stats(rows)
        print(
            "관용구 집중(세트 '간') "
            + " · ".join(f"…{t} {n}/{ts['n_cards']}장" for t, n in ts["top"])
        )
        print(
            f"표현 다양성 꼬리 {ts['n_tails']}종 / 항목 {ts['n_items']}개 = {ts['diversity']:.0%}"
            "   ← 모델 비교는 이 값으로 한다"
        )
    if lat:
        lat = sorted(lat)
        print(f"지연 p50 {lat[len(lat) // 2]:.2f}s  p90 {lat[int(len(lat) * 0.9)]:.2f}s")
    if show:
        for r in rows:
            c = by_id.get(r["case_id"])
            if not c:
                continue
            s = score(task, c, r["items"], r["error"], lexicon, ver)
            flags = "".join(k for k in "VTGDFX" if s[k]) or "-"
            print(f"\n[{r['case_id']}] {c.get('site') or ''} {c.get('from', '')}  flags={flags}")
            if r["error"]:
                print("   ERR", r["error"])
                continue
            inp = _norm(input_text(task, c))
            sents = input_sentences(task, c)
            seen: set[str] = set()
            for it in r["items"]:
                nt = _norm(it["text"])
                bad = nt in seen or (
                    it["source"] != "patient_message" and not form_ok(task, it["text"], inp, sents)
                )
                seen.add(nt)
                sp = (
                    not bad
                    and it["source"] != "general"
                    and any(
                        t not in STOP and traces_to_input(t, inp)
                        for t in _content_tokens(it["text"])
                    )
                )
                mark = "✗" if bad else ("●" if sp else "○")
                print(f"   {mark} {it['text']}   ← {it['source']}")


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    a = argparse.ArgumentParser()
    a.add_argument("--task", choices=["questions", "todos"], required=True)
    a.add_argument("--provider", choices=["anthropic", "openai", "google", "local", "bedrock"])
    a.add_argument("--model")
    a.add_argument(
        "--budget", type=float, default=float(os.environ.get("MEDIMATE_EVAL_BUDGET_USD", "0.50"))
    )
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--report", type=Path)
    a.add_argument("--show", action="store_true", help="항목 전부 출력(사람 판정용)")
    a.add_argument("--yes", action="store_true")
    a.add_argument("--prompt-version", help="questions-v1~v5, todos-v1|v2. 기본은 최신")
    args = a.parse_args()
    cases = load_inputs(args.task)
    version = args.prompt_version or (
        ap.QUESTIONS_VERSION if args.task == "questions" else ap.TODOS_VERSION
    )
    vsuf = "" if version.endswith("-v1") else "-" + version.split("-")[-1]
    if args.report:
        rows = [json.loads(ln) for ln in args.report.read_text(encoding="utf-8").splitlines() if ln]
        if n := reparse(rows):
            print(f"원문에서 다시 파싱: {n}건 (저장 당시 파싱 실패)")
        report(args.task, rows, cases, args.show)
        return
    if args.dry_run:
        rows = run(args.task, DryRun(), cases, version, RESULTS / f"{args.task}-dry-run.jsonl")
        report(args.task, rows, cases, args.show)
        return
    if not (args.provider and args.model):
        a.error("--provider 와 --model 필요 (또는 --dry-run / --report)")
    pi, po = require_price(args.model)  # 가격표에 없으면 호출 전에 멈춘다
    n = len(cases)
    est_in, est_out = (1200, 160) if args.task == "questions" else (900, 120)
    print(
        f"{args.task} · {args.model}: 호출 {n}회, 예상 비용 약 ${(n * est_in * pi + n * est_out * po) / 1e6:.3f}"
    )
    if not args.yes and input("진행? [y/N] ").strip().lower() != "y":
        return
    llm = LLMExtractor(args.provider, args.model, budget_usd=args.budget)
    rows = run(
        args.task,
        llm,
        cases,
        version,
        RESULTS / f"{args.task}-{safe_model_name(args.model)}{vsuf}.jsonl",
    )
    report(args.task, rows, cases, args.show)
    print(f"\n실제 비용 ${llm.usage.cost_usd(args.model):.3f}")


if __name__ == "__main__":
    main()
