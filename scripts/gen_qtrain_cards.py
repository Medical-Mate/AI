"""질문 후보 파인튜닝 파일럿 — 학습용 가상 카드를 만든다(Nova Pro, 2026-10-08).

평가 카드 100장(`evals/previsit_cards.jsonl`)은 학습에 쓰지 않는다. 여기서 만든 카드는 `TR` 번호를 달고
`evals/results/qtrain/`(gitignore)에 쌓인다. 정답 질문(Nova v8)과 거르기는 다음 단계에서 한다.

두 갈래
- seeds: 사용자 본인 테스트 카드 6장(운영 Langfuse, 사용자 확인 2026-10-08)의 짧고 거친 말투를 유지하고 부위를 바꾼 변형
- fresh: 진료 구역 25곳 + 전신·피부를 고르게 돌며 새로 만든 카드. 씨앗과 채팅 케이스의 값을 말투 참고로만 준다

  uv run --no-sync python scripts/gen_qtrain_cards.py --mode fresh --n 5 --dry-run   # 호출 0, 예상 비용
  uv run --no-sync python scripts/gen_qtrain_cards.py --mode fresh --n 5 --yes
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from medimate.llm.base import parse_json_text  # noqa: E402
from medimate.llm.providers import LLMExtractor  # noqa: E402
from medimate.ontology import load_ontology  # noqa: E402

MODEL = "apac.amazon.nova-pro-v1:0"  # 기본. 샘플 3차에서 Nova는 힌트를 값에 그대로 옮겼다 — --model로 바꿔 쓴다
OUT = Path("evals/results/qtrain")
SEEDS = Path("evals/results/qtrain/seeds.json")  # 사용자 테스트 6장(운영 Langfuse에서 뽑음, gitignore)
CARDS_PER_CALL = 5
EST_IN, EST_OUT = 1500, 900  # 호출당 토큰 어림(5장)
SEVERITY = ["1 (가벼운 불편)", "2 (신경 쓰여요)", "3 (꽤 아파요)", "4 (매우 심함)"]
# 카드마다 다르게 주는 상황 힌트·경과 방향 — 같은 요청 안에서 값이 한쪽으로 몰리지 않게(샘플 1차: 경과 4/6이 "점점 심해짐")
SITUATIONS = [
    "등산 다음 날", "장시간 운전 뒤", "감기 끝무렵", "새 약 먹기 시작한 뒤", "아이를 안다가", "회식 다음 날",
    "컴퓨터 오래 한 뒤", "운동 처음 시작하고", "잠을 잘못 자고", "넘어진 뒤", "매운 거 먹고", "날이 추워지고",
    "스트레스 많은 주", "오래 서서 일하고", "수영 다녀와서", "화장품 바꾸고", "렌즈 오래 끼고", "특별한 계기 없이",
    "여행 다녀와서", "밤샘 뒤", "커피 많이 마신 뒤", "무거운 것 들고", "부딪힌 뒤", "식사 거르고",
]
COURSES = ["나아지는 쪽", "비슷한 쪽", "심해지는 쪽", "좋았다 나빴다", "비움"]
# 느낌 계열 — 샘플 2차에서 한 요청 5장 중 3~4장이 "따가움"으로 몰렸다. 부위에 안 맞으면 모델이 가까운 말로 바꾼다
FEELS = [
    "욱신", "찌릿", "뻐근", "쓰림", "묵직", "콕콕", "화끈", "저림", "가려움", "답답", "시큰", "쑤심",
    "결림", "당김", "먹먹", "따끔", "울렁", "붓는 느낌", "뜨끈", "조이는 느낌",
]
# 복용약·지병·알러지는 여기서 고른다 — 모델에 맡기면 목록 앞쪽(혈압약·고혈압·페니실린)만 골랐다
MEDS = ["혈압약", "당뇨약", "진통제", "영양제", "갑상선약", "위장약", "콜레스테롤약", "수면제", "비타민", "피임약", "철분제"]
CONDS = ["고혈압", "당뇨", "천식", "갑상선", "위염", "비염", "고지혈증"]
ALLERGIES = ["페니실린", "아스피린", "갑각류", "꽃가루", "땅콩", "먼지"]

SYSTEM = """너는 진료 전 문답 앱의 **가상 테스트 카드**를 만든다. 실제 환자가 앱 채팅 문답에 답한 뒤 카드에 남은 값처럼 쓴다.

말투 (규칙만 따르고, 이 문장들을 값으로 옮기지 않는다)
- **값 길이는 카드마다 요청이 정한다.** "아주 짧게"면 값 하나가 2~6자다("뻐근해", "저려", "어제 갑자기", "비슷함" — 실제 사용자가 대부분 이렇다).
  "보통"이면 4~12자. 어느 쪽이든 문장으로 길게 쓰지 않는다. 띄어쓰기를 틀리거나 반말·존댓말이 섞여도 된다
- 의학 용어로 바꿔 쓰지 않는다. 환자가 느낀 말로 쓴다
- **카드마다 값을 새로 쓴다.** 요청에 나온 씨앗 카드나 다른 카드의 값을 그대로 옮기지 않는다

내용
- 요청이 카드마다 주는 부위·상황·경과 방향·느낌 계열로 **그 부위에 자연스러운 증상**을 쓴다
- **상황은 배경이다. 값에 그대로 옮기지 않는다.** 시작 칸에는 "언제부터"를 쓴다("3일 전", "그저께부터", "한 2주?", "어제 저녁"). 원인은 가끔만 덧붙인다
- **경과 방향도 환자 말로 바꿔 쓴다**("어제보단 나음", "그대로임", "더 아픈듯", "괜찮다가 또 아픔"). "비움"이면 null
- 느낌 계열은 출발점일 뿐이다. 같은 낱말을 그대로 쓰지 말고 환자가 할 법한 말로("욱신거려", "찌릿찌릿함", "쓰려요")
- 시작·느낌은 대부분 채운다. 악화·완화와 동반 증상은 3장 중 2장꼴로 채운다. 퍼짐은 2장 중 1장꼴. 없다는 답("없음", "안 퍼짐")도 가끔
- 환자는 진단을 모른다. **병명을 값에 넣지 않는다**
- 복용약·기저질환·알러지는 요청이 준 목록을 그대로 넣는다(없으면 빈 목록)
- 심각도는 요청이 준 값을 그대로 쓴다(null이면 null)

출력은 JSON 하나: {"cards":[{"site":"...","axes":{"onset":...,"character":...,"severity":...,"time_course":...,"exacerbating":...,"radiation":...,"associated":...},"profile":{"medications":[],"conditions":[],"allergies":[]}}]}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "site": {"type": "string"},
                    "axes": {
                        "type": "object",
                        "properties": {
                            k: {"type": ["string", "null"]}
                            for k in ("onset", "character", "severity", "time_course",
                                      "exacerbating", "radiation", "associated")
                        },
                    },
                    "profile": {
                        "type": "object",
                        "properties": {
                            k: {"type": "array", "items": {"type": "string"}}
                            for k in ("medications", "conditions", "allergies")
                        },
                    },
                },
                "required": ["site", "axes", "profile"],
            },
        }
    },
    "required": ["cards"],
}


def zones() -> list[tuple[str, str]]:
    o = load_ontology()
    rows = []
    for aid in sorted(o.anchors):
        for z in o.zones(aid):
            n = o.get(z) if isinstance(z, str) else z
            rows.append((n.name_ko, n.laterality))
    return rows + [("전신", "none"), ("피부", "none")]


def seed_card(s: dict) -> str:
    # 트레이스 식별자 가림이 부위까지 {NAME}으로 바꾼 씨앗이 있다 — 자리표를 베끼지 않게 중립 표시로
    return s["user"].split("\n\n의사에게")[0].replace("{NAME}", "(가린 값)")


def seed_lines(seeds: list[dict]) -> str:
    return "\n\n".join(seed_card(s) for s in seeds)


def site_label(name: str, lat: str, rng: random.Random) -> str:
    if lat == "left_right":
        return f"{rng.choice(['왼쪽', '오른쪽'])} {name}"
    return name


def card_specs(picks: list[str], rng: random.Random) -> str:
    lines = []
    for j, site in enumerate(picks, 1):
        sev = rng.choice([None] + SEVERITY + SEVERITY[1:3])  # 2·3이 흔하다
        prof = {"medications": [], "conditions": [], "allergies": []}
        if rng.random() < 0.3:
            prof["medications"] = rng.sample(MEDS, rng.choice([1, 1, 2]))
            if rng.random() < 0.5:
                prof["conditions"] = rng.sample(CONDS, 1)
        if rng.random() < 0.12:
            prof["allergies"] = rng.sample(ALLERGIES, 1)
        # 실제 입력 6장은 값 중앙값 4자·75%가 6자 이하였다. 생성기는 길게 쓰는 쪽이라(Sonnet 12자) 절반을 짧게 못 박는다
        length = "아주 짧게" if rng.random() < 0.5 else "보통"
        lines.append(
            f"카드 {j}: 부위 {site} · 길이 {length} · 상황 {rng.choice(SITUATIONS)} · 경과 방향 {rng.choice(COURSES)}"
            f" · 느낌 계열 {rng.choice(FEELS)} · 심각도 {json.dumps(sev, ensure_ascii=False)}"
            f" · 복용약·지병·알러지 {json.dumps(prof, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def build_requests(mode: str, n_calls: int, rng: random.Random) -> list[str]:
    zs = zones()
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))
    reqs = []
    for i in range(n_calls):
        picks = [site_label(*zs[(i * CARDS_PER_CALL + j) % len(zs)], rng) for j in range(CARDS_PER_CALL)]
        specs = card_specs(picks, rng)
        if mode == "seeds":
            # 씨앗 하나만 준다. 말투(짧고 거친 정도)만 닮고 값은 새로 쓴다
            reqs.append(
                "씨앗 카드(말투만 참고, 값은 옮기지 말 것):\n"
                + seed_card(rng.choice(seeds))  # 순서대로 고르면 호출이 적을 때 앞의 씨앗만 쓰인다
                + f"\n\n씨앗과 같은 정도로 짧고 거친 말투로 {CARDS_PER_CALL}장:\n"
                + specs
            )
        else:
            reqs.append(f"새 카드 {CARDS_PER_CALL}장:\n" + specs)
    return reqs


CLI_TAIL = "\n\n위 형식의 JSON 하나만 출력한다. 설명·코드 펜스 없이."


def call_cli(backend: str, model: str | None, user: str, workdir: Path) -> str:
    """구독 CLI로 한 번 부른다(AWS 비용 0). 저장소 밖 빈 폴더에서 돌려 프로젝트 CLAUDE.md·메모리가 섞이지 않게 한다."""
    if backend == "claude":
        cmd = ["claude", "-p", "--system-prompt", SYSTEM, "--tools", "", "--no-session-persistence",
               "--output-format", "text"] + (["--model", model] if model else [])
        r = subprocess.run(cmd, input=user + CLI_TAIL, capture_output=True, text=True, encoding="utf-8",
                           cwd=workdir, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"claude 종료 {r.returncode}: {r.stderr[-300:]}")
        return r.stdout
    last = workdir / "codex-last.txt"
    cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-C", str(workdir),
           "-o", str(last)] + (["-m", model] if model else []) + ["-"]
    r = subprocess.run(cmd, input=SYSTEM + "\n\n" + user + CLI_TAIL, capture_output=True, text=True,
                       encoding="utf-8", timeout=900, shell=sys.platform == "win32")
    if r.returncode != 0:
        raise RuntimeError(f"codex 종료 {r.returncode}: {r.stderr[-300:]}")
    return last.read_text(encoding="utf-8")


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--mode", choices=["seeds", "fresh"], required=True)
    a.add_argument("--n", type=int, required=True, help="만들 카드 수(5장 단위로 올림)")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--backend", choices=["bedrock", "claude", "codex"], default="bedrock")
    a.add_argument("--model", default=None, help="bedrock은 가격표 ID(기본 Nova Pro), CLI는 그 CLI의 모델 이름(비우면 기본)")
    a.add_argument("--budget", type=float, default=0.5)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--yes", action="store_true")
    args = a.parse_args()
    n_calls = -(-args.n // CARDS_PER_CALL)
    # 모드마다 난수를 따로 — 같은 seed로 두 모드를 돌리면 부위·힌트가 같은 카드 쌍이 생겼다(샘플 2차).
    # 백엔드는 난수에 넣지 않는다 — 같은 seed면 모든 생성기가 같은 요청을 받아 나란히 비교된다
    rng = random.Random(f"{args.mode}-{args.seed}")
    reqs = build_requests(args.mode, n_calls, rng)
    model = args.model or (MODEL if args.backend == "bedrock" else None)
    if args.backend == "bedrock":
        from medimate.llm.providers import PRICES

        pi, po = PRICES[model]
        cost = f"예상 ${n_calls * (EST_IN * pi + EST_OUT * po) / 1e6:.3f} (상한 ${args.budget})"
    else:
        cost = "AWS 비용 0 (구독 사용량)"
    print(f"{args.mode} · {args.backend} {model or '(기본)'}: 호출 {n_calls}회 → 카드 약 {n_calls * CARDS_PER_CALL}장, {cost}")
    if args.dry_run:
        print("\n--- 첫 요청 ---\n" + reqs[0])
        return
    if not args.yes and input("진행? [y/N] ").strip().lower() != "y":
        return
    ex = LLMExtractor("bedrock", model, budget_usd=args.budget) if args.backend == "bedrock" else None
    workdir = Path(tempfile.mkdtemp(prefix="qtrain-"))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.backend if args.backend != "bedrock" else model.split(".")[-1].split(":")[0][:20]
    if args.backend != "bedrock" and model:
        tag += "-" + model
    out = OUT / f"cards-{args.mode}-s{args.seed}-{tag}.jsonl"
    k = 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for i, user in enumerate(reqs):
            try:
                if ex:
                    text, _, _ = ex.complete_json(SYSTEM, user, SCHEMA, max_tokens=2048)
                else:
                    text = call_cli(args.backend, model, user, workdir)
                cards = parse_json_text(text)["cards"]  # 코드 펜스로 감싸 보내기도 한다
            except Exception as e:  # noqa: BLE001
                print(f"  요청 {i} 실패: {type(e).__name__}: {str(e)[:200]}")
                continue
            for c in cards:
                k += 1
                c["id"] = f"TR-{args.mode[0].upper()}{args.seed}-{tag}-{k:04d}"
                c["generator"] = f"{args.backend}:{model or 'default'}"
                c.setdefault("axes", {})["site"] = c.get("site")
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    spent = f" · 실제 ${ex.usage.cost_usd(model):.4f}" if ex else ""
    print(f"카드 {k}장 → {out}{spent}")


if __name__ == "__main__":
    main()
