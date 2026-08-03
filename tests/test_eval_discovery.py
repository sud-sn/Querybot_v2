from pathlib import Path

from admin.routes import _eval_case_files


def test_eval_discovery_finds_all_supported_client_suites(tmp_path, monkeypatch):
    root = tmp_path / "evals" / "clients" / "tenant-a" / "domain"
    root.mkdir(parents=True)
    expected = {
        root / "golden_questions.yaml",
        root / "finance_questions.yml",
        root / "regression_cases.json",
        root / "eval_suite.yaml",
    }
    for path in expected:
        path.write_text("cases: []", encoding="utf-8")
    (root / "semantic_model.json").write_text("{}", encoding="utf-8")
    (root / "notes.yaml").write_text("notes: []", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    found = {path.resolve() for path in _eval_case_files("tenant-a")}

    assert found == {path.resolve() for path in expected}


def test_eval_discovery_is_scoped_to_requested_client(tmp_path, monkeypatch):
    other = tmp_path / "evals" / "clients" / "tenant-b"
    other.mkdir(parents=True)
    (other / "other_questions.yaml").write_text("cases: []", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    assert _eval_case_files("tenant-a") == []
