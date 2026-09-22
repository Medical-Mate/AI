"""맞춤법 교정 가드 — 생성은 모델이, 검증은 코드가. 실호출 없음."""

from medimate.text.spell import MAX_JAMO_EDITS, edit_distance, guard, jamo


def test_jamo_decomposition_and_distance():
    assert jamo("왓") == "ㅇㅘㅅ" and jamo("왔") == "ㅇㅘㅆ"
    assert edit_distance(jamo("나왓지만"), jamo("나왔지만")) == 1
    assert edit_distance(jamo("괜찬대요"), jamo("괜찮대요")) == 1
    assert edit_distance(jamo("낫으면"), jamo("나으면")) <= MAX_JAMO_EDITS


def test_accepts_small_spelling_fixes():
    r = guard("간수치가 조금 높게 나왓지만 괜찬대요.", "간수치가 조금 높게 나왔지만 괜찮대요.")
    assert r.accepted and r.ops == [("나왓지만", "나왔지만"), ("괜찬대요.", "괜찮대요.")]
    r = guard("안 낫으면 2주뒤 오래요.", "안 나으면 2주뒤 오래요.")
    assert r.accepted and r.text == "안 나으면 2주뒤 오래요."


def test_rejects_meaning_changes():
    # 낱말을 바꿈 — 어절 거리가 크다
    r = guard("소염제 하루 두 번 일주일.", "진통제 하루 두 번 일주일.")
    assert not r.accepted and r.text == "소염제 하루 두 번 일주일."
    # 문장을 다듬음(길이 변화 + 거리)
    r = guard("약은 저녁에 한 알 먹고", "저녁에 약을 한 알 복용하세요")
    assert not r.accepted


def test_rejects_digit_changes():
    r = guard("2주 뒤에 오래요", "3주 뒤에 오래요")
    assert not r.accepted and r.reason == "digits_changed"


def test_rejects_when_protected_term_is_split_or_lost():
    r = guard("위산약 2주치 받고", "위산 약 2주치 받고", protected=["위산약"])
    # 공백만 달라진 것은 용어 검사에서 공백을 무시하므로 통과한다 — 뜻이 같다
    assert r.accepted
    r = guard("위산약 2주치 받고", "제산제 2주치 받고", protected=["위산약"])
    assert not r.accepted and r.reason.startswith("term_lost")


def test_unchanged_or_empty_is_not_accepted():
    assert guard("감기래요", "감기래요").reason == "unchanged"
    assert guard("감기래요", "").reason == "empty"


def test_spacing_only_change_is_accepted():
    r = guard("새약은반알부터먹고", "새 약은 반 알부터 먹고")
    assert r.accepted and r.text == "새 약은 반 알부터 먹고"
