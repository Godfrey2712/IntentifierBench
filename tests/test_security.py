import os
import sys
import importlib

import pytest


def _reload_app_with_skip_env(docs_path, monkeypatch):
    """Reload the app module with SKIP_MODEL_LOAD=1 so tests don't load heavy models."""
    monkeypatch.setenv('SKIP_MODEL_LOAD', '1')

    # Ensure a fresh import of app
    if 'app' in sys.modules:
        del sys.modules['app']

    app_mod = importlib.import_module('app')
    # Override DOCUMENTS_PATH to our temp docs directory
    app_mod.DOCUMENTS_PATH = str(docs_path)
    return app_mod


def test_path_traversal_rejected(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "known1.txt").write_text("known content")

    app_mod = _reload_app_with_skip_env(docs, monkeypatch)
    client = app_mod.app.test_client()

    resp = client.post('/identify', data={'unknown_doc': '../../etc/passwd', 'selected_model': 'model1'})
    assert resp.status_code == 400
    assert resp.is_json
    assert 'error' in resp.get_json()


def test_nonexistent_filename_rejected(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "known1.txt").write_text("known content")

    app_mod = _reload_app_with_skip_env(docs, monkeypatch)
    client = app_mod.app.test_client()

    resp = client.post('/identify', data={'unknown_doc': 'no_such_file.txt', 'selected_model': 'model1'})
    assert resp.status_code == 400
    assert resp.is_json
    assert 'error' in resp.get_json()
