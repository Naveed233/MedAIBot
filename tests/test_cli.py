from medaibot.cli import build_parser


def test_serve_port_comes_from_the_port_env_var(monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    assert build_parser().parse_args(["serve"]).port == 10000
    monkeypatch.delenv("PORT")
    assert build_parser().parse_args(["serve"]).port == 8000
