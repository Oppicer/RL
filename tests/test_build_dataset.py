import subprocess
import os
from pathlib import Path

def test_build_dataset(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = repo / 'data' / 'build_dataset.sh'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo / 'tests' / 'fake_modules')
    subprocess.check_call(['bash', str(script)], cwd=tmp_path, env=env)

    dataset_dir = tmp_path / 'dataset'
    sample = dataset_dir / 'sample.txt'
    tokens = dataset_dir / 'sample.tok'
    assert sample.exists()
    assert tokens.exists()
    assert sample.stat().st_size > 0
    assert tokens.stat().st_size > 0
