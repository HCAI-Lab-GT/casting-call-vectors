import threading

import pandas as pd
import pytest

from pvx.experiments.gold_prompt_experiments import GoldPromptExperiments

COLUMNS = [
    "role", "layer", "sample_count", "alpha", "temperature", "question",
    "assistant_axis", "baseline", "steered",
    "assistant_axis_score", "baseline_score", "steered_score",
]


@pytest.fixture
def exp():
    e = object.__new__(GoldPromptExperiments)
    e.result_columns = COLUMNS
    return e


def _row(question, alpha=2.0, steered="s"):
    return pd.DataFrame(
        [{
            "role": "lawyer", "layer": 16, "sample_count": 40, "alpha": alpha,
            "temperature": 0.2, "question": question, "assistant_axis": "a",
            "baseline": "b", "steered": steered, "assistant_axis_score": 1,
            "baseline_score": 2, "steered_score": 3,
        }]
    )


def test_merge_creates_header_only_file(exp, tmp_path):
    path = tmp_path / "r.csv"
    exp._merge_into_csv(path, exp._empty_results_df())
    assert list(pd.read_csv(path).columns) == COLUMNS
    assert pd.read_csv(path).empty


def test_concurrent_writers_lose_no_rows(exp, tmp_path):
    path = tmp_path / "r.csv"

    def worker(w):
        for i in range(10):
            exp._merge_into_csv(path, _row(f"w{w}-q{i}"))

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(pd.read_csv(path)) == 40


def test_duplicate_key_keeps_existing_row(exp, tmp_path):
    path = tmp_path / "r.csv"
    exp._merge_into_csv(path, _row("q", steered="first"))
    merged = exp._merge_into_csv(path, _row("q", steered="second"))
    assert len(merged) == 1
    assert pd.read_csv(path)["steered"].tolist() == ["first"]


def test_stale_in_memory_frame_does_not_clobber_other_jobs(exp, tmp_path):
    path = tmp_path / "r.csv"
    stale = exp._merge_into_csv(path, _row("mine"))
    exp._merge_into_csv(path, _row("theirs"))  # another job writes meanwhile
    exp._merge_into_csv(path, stale)  # end-of-role write from our stale copy
    assert sorted(pd.read_csv(path)["question"]) == ["mine", "theirs"]


def test_no_temp_files_besides_lockfile(exp, tmp_path):
    path = tmp_path / "r.csv"
    exp._merge_into_csv(path, _row("q"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.csv", "r.csv.lock"]
