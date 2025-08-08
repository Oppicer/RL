import sys
import types
import importlib
import importlib.util
from pathlib import Path
from unittest import mock
import pytest


def import_with_stubs(monkeypatch):
    # Remove cached module if present
    if 'pipelines.train_full' in sys.modules:
        del sys.modules['pipelines.train_full']

    # Stub heavy dependencies
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))

    tfm = types.ModuleType('transformers')
    tfm.AutoTokenizer = object
    tfm.AutoModelForCausalLM = object
    tfm.Trainer = object
    tfm.TrainingArguments = object
    monkeypatch.setitem(sys.modules, 'transformers', tfm)

    dsets = types.ModuleType('datasets')
    dsets.load_dataset = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'datasets', dsets)

    peft = types.ModuleType('peft')
    peft.LoraConfig = object
    peft.get_peft_model = lambda m, c: m
    monkeypatch.setitem(sys.modules, 'peft', peft)

    # Import the module from its path to avoid package issues
    path = Path(__file__).resolve().parents[1] / 'pipelines' / 'train_full.py'
    spec = importlib.util.spec_from_file_location('train_full', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_argument_parsing(monkeypatch):
    mod = import_with_stubs(monkeypatch)

    with mock.patch.object(sys, 'argv', ['prog', '--config', 'cfg.yaml', '--export_fp16', '--export_awq', '--no_dgm']):
        args = mod.parse_args()
    assert args.config == 'cfg.yaml'
    assert args.export_fp16 is True
    assert args.export_awq is True
    assert args.no_dgm is True

    with mock.patch.object(sys, 'argv', ['prog']):
        with pytest.raises(SystemExit):
            mod.parse_args()
