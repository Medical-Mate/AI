"""Bedrock 실제 지출 조회 — AWS가 센 토큰으로 계산한다.

우리 러너가 찍는 비용은 **우리가 센 토큰**이다. AWS가 실제로 얼마를 뺐는지는 별개이고,
크레딧으로 돌릴 때 그게 안 보이면 실호출을 겁내게 된다. 이 스크립트가 그 간격을 메운다.

Cost Explorer를 쓰지 않는다. 신규 계정은 데이터 수집에 하루가 걸리고(`DataUnavailableException`),
요청당 $0.01이 붙는다. 대신 **CloudWatch `AWS/Bedrock` 토큰 지표**를 쓴다 —
몇 분 지연, 사실상 무료, 모델별로 나온다. 단가는 `providers.PRICES` 한 곳에서 가져온다.

    uv run python scripts/bedrock_spend.py
    uv run python scripts/bedrock_spend.py --days 1 --credit 100

주의: CloudWatch는 **그 리전에서 일어난 모든 호출**을 센다. eval뿐 아니라 손으로 찔러 본
호출·스모크 테스트도 들어간다. 그래서 우리 러너 합계보다 조금 크게 나오는 게 정상이다.
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "src")

from medimate.llm.providers import PRICES  # noqa: E402

METRICS = ("InputTokenCount", "OutputTokenCount", "Invocations")


def fetch(cw, model_id: str, start, end) -> dict[str, int]:
    out = {}
    for m in METRICS:
        r = cw.get_metric_statistics(
            Namespace="AWS/Bedrock",
            MetricName=m,
            Dimensions=[{"Name": "ModelId", "Value": model_id}],
            StartTime=start,
            EndTime=end,
            Period=86400,
            Statistics=["Sum"],
        )
        out[m] = int(sum(d["Sum"] for d in r["Datapoints"]))
    return out


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--days", type=int, default=30)
    a.add_argument("--region", default="ap-northeast-2")
    a.add_argument("--credit", type=float, default=100.0, help="받은 크레딧. 남은 양 계산용")
    args = a.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    import boto3

    cw = boto3.client("cloudwatch", region_name=args.region)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    # 가격표에 있는 Bedrock 모델만 본다(= 우리가 쓸 수 있는 것). 나머지는 조회할 이유가 없다
    ids = [m for m in PRICES if "." in m and ("amazon" in m or "anthropic" in m)]

    print(f"리전 {args.region} · 최근 {args.days}일 · AWS가 센 토큰 기준\n")
    print(f"{'모델':52}{'호출':>6}{'입력 tok':>11}{'출력 tok':>10}{'비용':>10}")
    print("-" * 89)
    total = 0.0
    for mid in sorted(ids):
        try:
            d = fetch(cw, mid, start, end)
        except Exception as e:  # noqa: BLE001 — 권한·리전 문제면 그 모델만 건너뛴다
            print(f"{mid:52}  조회 실패: {type(e).__name__}")
            continue
        if not d["Invocations"]:
            continue
        pi, po = PRICES[mid]
        cost = (d["InputTokenCount"] * pi + d["OutputTokenCount"] * po) / 1e6
        total += cost
        print(
            f"{mid:52}{d['Invocations']:>6}{d['InputTokenCount']:>11,}"
            f"{d['OutputTokenCount']:>10,}{cost:>10.4f}"
        )
    print("-" * 89)
    print(f"{'합계':52}{'':>27}{total:>10.4f}")
    print(
        f"\n크레딧 ${args.credit:.0f} 중 **${total:.2f} 사용 ({total / args.credit:.2%})** · "
        f"남음 약 ${args.credit - total:.2f}"
    )
    print("CloudWatch는 eval 외 호출(손으로 찔러 본 것·스모크)도 센다. 러너 합계보다 크면 정상.")


if __name__ == "__main__":
    main()
