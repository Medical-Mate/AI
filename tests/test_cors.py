"""브라우저가 우리를 직접 부를 때만 필요하다. 백엔드가 프록시하면 CORS는 개입하지 않는다.

아직 안 정해졌으므로 **env 하나로 켤 수 있게만** 해 둔다. 기본은 꺼짐 — 붙이지 않은 것과
동작이 같아야 한다.
"""

import pytest
from fastapi.testclient import TestClient

from medimate.api.app import CORS_ENV, create_app

WEB = "https://demo.example.com"


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(CORS_ENV, raising=False)
    c = TestClient(create_app())
    r = c.get("/health", headers={"Origin": WEB})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
    assert r.json()["cors_origins"] is None


def test_listed_origin_is_allowed(monkeypatch):
    monkeypatch.setenv(CORS_ENV, f"{WEB}, https://www.example.com")
    c = TestClient(create_app())
    r = c.get("/health", headers={"Origin": WEB})
    assert r.headers["access-control-allow-origin"] == WEB
    assert r.json()["cors_origins"] == 2


def test_unlisted_origin_gets_no_header(monkeypatch):
    """헤더가 없으면 브라우저가 응답을 스크립트에 넘기지 않는다"""
    monkeypatch.setenv(CORS_ENV, WEB)
    c = TestClient(create_app())
    r = c.get("/health", headers={"Origin": "https://elsewhere.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_preflight_allows_the_signature_headers(monkeypatch):
    """프록시 없이 직접 부르면 HMAC 헤더가 프리플라이트를 지나야 한다"""
    monkeypatch.setenv(CORS_ENV, WEB)
    c = TestClient(create_app())
    r = c.options(
        "/v1/previsit/turns",
        headers={
            "Origin": WEB,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-signature,x-timestamp,x-request-id",
        },
    )
    assert r.status_code == 200
    allowed = r.headers["access-control-allow-headers"].lower()
    for h in ("x-signature", "x-timestamp", "x-request-id"):
        assert h in allowed, h


def test_star_is_refused(monkeypatch):
    """브라우저에 HMAC 시크릿을 둘 수 없어 출처 목록이 유일한 문지기가 된다"""
    monkeypatch.setenv(CORS_ENV, "*")
    with pytest.raises(ValueError, match=CORS_ENV):
        create_app()
    monkeypatch.setenv(CORS_ENV, f"{WEB},*")
    with pytest.raises(ValueError, match=CORS_ENV):
        create_app()
