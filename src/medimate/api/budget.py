"""프로세스 전역 일일 LLM 지출 상한.

**왜 요청 단위 상한으로는 부족한가.** 러너의 `--budget`은 `LLMExtractor` 인스턴스에 붙어 있는데
API는 요청마다 새 인스턴스를 만든다(`app._default_factory`). 그래서 그 상한은 **한 요청 안에서만**
산다 — 요청이 1,000번 오면 상한도 1,000번 새로 시작한다. 사비·크레딧으로 도는 공개 데모에는
누적을 보는 것이 따로 필요하다.

**끄는 것이 기본이다.** `MEDIMATE_DAILY_BUDGET_USD`가 없으면 아무 일도 하지 않는다 —
로컬·테스트·eval 러너가 이 파일 때문에 달라지면 안 된다. 운영 env에서만 켠다.

**한계를 알고 둔다.** 카운터는 메모리에 있어 **컨테이너를 재시작하면 0이 된다.** 재시작해도
남는 저장은 지금 만들지 않는다 — 우리 쪽 상한은 "실수로 새는 것"을 막는 장치이고, 진짜 상한은
클라우드 쪽 예산 알림이다. 둘이 같은 층에 있으면 하나가 죽었을 때 둘 다 죽는다.

그리고 여기서 세는 것은 **우리 가격표 기준 추정치**다(`providers.PRICES`). 청구서와 다를 수
있으므로 `/health`의 `llm_budget`으로 그대로 내보인다 — 켠 줄 알았는데 안 켜진 상태,
상한이 생각과 다른 상태를 밖에서 볼 수 있어야 한다(`hmac_required`와 같은 이유).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from medimate.llm.providers import BudgetExceeded

ENV_CAP = "MEDIMATE_DAILY_BUDGET_USD"
ENV_TZ = "MEDIMATE_BUDGET_RESET_TZ"

# 하루의 경계를 어느 시간대로 볼 것인가. **기본이 UTC가 아니다.**
# UTC 자정은 KST 오전 9시다 — 데모 날 오후에 상한이 차면 **다음 날 아침까지 안 풀린다.**
# 상한은 새는 것을 막는 장치이지 하루를 날리는 장치가 아니므로, 사람이 쓰는 시간대에 맞춘다.
DEFAULT_TZ = "Asia/Seoul"


def _tz() -> ZoneInfo:
    name = (os.getenv(ENV_TZ) or "").strip() or DEFAULT_TZ
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — 오타로 조용히 UTC가 되면 리셋 시각이 9시간 어긋난다
        raise ValueError(
            f"{ENV_TZ}={name!r} — 알 수 없는 시간대입니다(예: Asia/Seoul, UTC)"
        ) from None


class DailyBudgetExceeded(BudgetExceeded):
    """`BudgetExceeded`를 물려받는다 — 기존 핸들러가 이미 503으로 바꾼다.

    503인 이유: 예산 소진은 **서버 상태**지 요청의 잘못이 아니다. 같은 요청을 내일 보내면
    통과한다. 402는 결제 문제를, 429는 이 클라이언트가 많이 보냈음을 뜻하는데 둘 다 사실이
    아니다. 503이면 클라이언트가 재시도 백오프를 걸기도 좋다.
    """


def _today() -> date:
    return datetime.now(_tz()).date()


def _next_midnight() -> str:
    """다음 리셋 시각. **그 시간대의 오프셋이 붙은 ISO**로 낸다 — 백엔드가 눈으로 읽는 값이다."""
    tomorrow = _today() + timedelta(days=1)
    return datetime.combine(tomorrow, time.min, tzinfo=_tz()).isoformat()


@dataclass
class DailyBudget:
    """리셋 시간대(기본 `Asia/Seoul`) 자정 기준 당일 누적. `cap_usd`가 None이면 꺼진 상태."""

    cap_usd: float | None = None
    _day: date = field(default_factory=_today)
    _spent: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_env(cls) -> DailyBudget:
        raw = (os.getenv(ENV_CAP) or "").strip()
        if not raw:
            return cls(cap_usd=None)
        try:
            cap = float(raw)
        except ValueError:
            # 오타로 조용히 꺼지면 안 된다 — 켤 의도가 있었다는 게 값의 존재로 드러난다
            raise ValueError(f"{ENV_CAP}={raw!r} — 숫자여야 합니다(예: 5.0)") from None
        if cap <= 0:
            raise ValueError(f"{ENV_CAP}={raw!r} — 0보다 커야 합니다. 끄려면 변수를 비우세요")
        return cls(cap_usd=cap)

    @property
    def enabled(self) -> bool:
        return self.cap_usd is not None

    def _roll(self) -> None:
        """날이 바뀌었으면 0으로. 호출자가 락을 잡고 있어야 한다."""
        today = _today()
        if today != self._day:
            self._day, self._spent = today, 0.0

    def check(self) -> None:
        """LLM을 부르기 **직전에** 부른다. 상한에 닿았으면 올린다."""
        if not self.enabled:
            return
        with self._lock:
            self._roll()
            if self._spent >= self.cap_usd:  # type: ignore[operator]
                raise DailyBudgetExceeded(
                    f"일일 LLM 예산 소진(${self._spent:.4f} / ${self.cap_usd}), "
                    f"{_next_midnight()}에 초기화"
                )

    def record(self, cost_usd: float) -> None:
        """호출 **뒤에** 실제 사용량으로. 음수·0은 무시한다."""
        if not self.enabled or cost_usd <= 0:
            return
        with self._lock:
            self._roll()
            self._spent += cost_usd

    def status(self) -> dict[str, Any] | None:
        """`/health`용. 꺼져 있으면 None."""
        if not self.enabled:
            return None
        with self._lock:
            self._roll()
            return {
                "cap_usd": self.cap_usd,
                "spent_today_usd": round(self._spent, 6),
                "resets_at": _next_midnight(),
            }


class BudgetGuarded:
    """LLM을 부르는 객체를 감싸 **호출 직전 확인 · 직후 기록**을 붙인다.

    턴 종류를 미리 알아내 거르지 않는 이유: 어떤 턴이 LLM을 부르는지는 엔진 안에서 갈린다
    (선택지만 온 턴, 폰이 뽑아 온 턴, 이미 끝난 세션의 턴은 안 부른다). 밖에서 추측하면
    **LLM을 안 쓸 턴을 막는 쪽으로** 틀리고, 그건 예산이 죽어도 문진은 이어지게 하려는
    설계와 정면으로 어긋난다. 실제로 부르는 자리에 붙이면 추측이 필요 없다.
    """

    _WRAPPED = ("extract", "extract_raw", "complete_json", "classify")

    def __init__(self, inner: Any, budget: DailyBudget):
        self._inner = inner
        self._budget = budget

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name not in self._WRAPPED or not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            self._budget.check()
            before = self._cost()
            try:
                return attr(*args, **kwargs)
            finally:
                # 예외로 끝나도 쓴 만큼은 센다 — 파싱 실패한 호출도 요금은 나간다
                self._budget.record(self._cost() - before)

        return call

    def _cost(self) -> float:
        """지금까지 이 객체가 쓴 비용. 사용량을 안 들고 있으면 0(테스트 대역 등)."""
        usage = getattr(self._inner, "usage", None)
        model = getattr(self._inner, "model_id", None)
        if usage is None or model is None or not hasattr(usage, "cost_usd"):
            return 0.0
        try:
            return float(usage.cost_usd(model))
        except Exception:  # noqa: BLE001 — 비용 계산 실패로 요청을 죽이지 않는다
            return 0.0
