import pytest

from care_analytics.generate import GenConfig, generate
from care_analytics.paths import get_paths
from care_analytics.quality import connect, run_quality
from care_analytics.warehouse import build_warehouse


@pytest.fixture(scope="session")
def paths(tmp_path_factory):
    """A small end-to-end dataset: generated, quality-checked and loaded into the warehouse."""
    p = get_paths(tmp_path_factory.mktemp("data"))
    generate(p, GenConfig(scale=0.08, seed=11))
    run_quality(p)
    build_warehouse(p)
    return p


@pytest.fixture()
def con(paths):
    c = connect(paths)
    yield c
    c.close()
