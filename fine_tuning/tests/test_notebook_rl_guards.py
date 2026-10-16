"""CPU audit checks in isolated processes; no teacher SDK or external calls."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class GuardTests(unittest.TestCase):
    def run_guard(self, code, directory=None):
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            root = Path(tmp)
            (root / "benchmark").mkdir()
            (root / "benchmark" / "data").write_text("benchmark")
            (root / "private").write_text("private")
            (root / "allowed").write_text("allowed")
            prelude = """
import os, sys
from pathlib import Path
from fine_tuning.notebook_rl.guards import install_guards, assert_no_teacher_imports
r = Path(sys.argv[1])
def blocked(action, message):
    try:
        action()
    except PermissionError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError('open/import was not blocked')
"""
            result = subprocess.run([sys.executable, "-c", prelude + code, tmp],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_protected_paths_and_allowed_file(self):
        self.run_guard("""
install_guards(forbidden_roots=[r/'benchmark'], vault_files=[r/'private'],
               allow_vault=False, allow_benchmark=False)
blocked(lambda: (r/'benchmark'/'data').read_text(), 'benchmark path blocked')
blocked(lambda: (r/'private').read_bytes(), 'vault path blocked')
assert (r/'allowed').read_text() == 'allowed'
blocked(lambda: os.open(r/'private', os.O_RDONLY), 'vault path blocked')
blocked(lambda: (r/'benchmark'/'new').open('w'), 'benchmark path blocked')
""")

    def test_symlink_parent_traversal_bytes_and_fd_aliases(self):
        self.run_guard("""
(r/'link').symlink_to(r/'benchmark', target_is_directory=True)
(r/'private_link').symlink_to(r/'private')
fd = os.open(r/'private', os.O_RDONLY)
install_guards(forbidden_roots=[r/'benchmark'], vault_files=[r/'private'],
               allow_vault=False, allow_benchmark=False)
blocked(lambda: (r/'link'/'data').read_bytes(), 'benchmark path blocked')
blocked(lambda: (r/'private_link').read_bytes(), 'vault path blocked')
blocked(lambda: open(os.fsencode(r/'private'), 'rb'), 'vault path blocked')
blocked(lambda: (r/'allowed'/'..'/'private').open('rb'), 'vault path blocked')
blocked(lambda: open(fd, 'rb', closefd=False), 'vault path blocked')
blocked(lambda: Path(f'/proc/self/fd/{fd}').read_bytes(), 'vault path blocked')
os.close(fd)
""")

    def test_hardlink_aliases_fail_closed(self):
        self.run_guard("""
os.link(r/'private', r/'alias')
os.link(r/'benchmark'/'data', r/'benchmark_alias')
install_guards(forbidden_roots=[r/'benchmark'], vault_files=[r/'private'],
               allow_vault=False, allow_benchmark=False)
blocked(lambda: (r/'alias').read_bytes(), 'hard-linked')
blocked(lambda: (r/'benchmark_alias').read_bytes(), 'hard-linked')
""")

    def test_missing_path_symlink_loop_fails_closed(self):
        self.run_guard("""
(r/'loop').symlink_to(r/'loop')
install_guards(forbidden_roots=[r/'benchmark'], vault_files=[r/'private'],
               allow_vault=False, allow_benchmark=False)
blocked(lambda: (r/'loop').read_bytes(), 'cannot be checked')
# CPython rejects NUL before emitting an open audit event.
try:
    open(str(r/'allowed') + '\\x00', 'rb')
except ValueError:
    pass
else:
    raise AssertionError('NUL path accepted')
""")

    def test_permissions_are_separate(self):
        self.run_guard("""
install_guards(forbidden_roots=[r/'benchmark'], vault_files=[r/'private'],
               allow_vault=True, allow_benchmark=False)
assert (r/'private').read_text() == 'private'
blocked(lambda: (r/'benchmark'/'data').read_bytes(), 'benchmark path blocked')
""")
        self.run_guard("""
install_guards(forbidden_roots=[r/'benchmark'], vault_files=[r/'private'],
               allow_vault=False, allow_benchmark=True)
assert (r/'benchmark'/'data').read_text() == 'benchmark'
blocked(lambda: (r/'private').read_bytes(), 'vault path blocked')
""")

    def test_second_install_rejected(self):
        self.run_guard("""
install_guards(forbidden_roots=[], vault_files=[], allow_vault=False, allow_benchmark=False)
try:
    install_guards(forbidden_roots=[], vault_files=[], allow_vault=True, allow_benchmark=True)
except RuntimeError as exc:
    assert 'already installed' in str(exc)
else:
    raise AssertionError('second installation allowed')
""")

    def test_teacher_import_and_loaded_descendants(self):
        self.run_guard("""
install_guards(forbidden_roots=[], vault_files=[], allow_vault=False, allow_benchmark=False)
for name in ('openai', 'anthropic', 'google.generativeai', 'google.genai', 'cohere', 'mistralai'):
    blocked(lambda: __import__(name), 'teacher import blocked')
    assert name not in sys.modules
assert_no_teacher_imports()
sys.modules['openai.fake'] = object()
try:
    assert_no_teacher_imports()
except RuntimeError:
    pass
else:
    raise AssertionError('loaded descendant accepted')
""")

    def test_install_rejects_preloaded_teacher_and_nonboolean_policy(self):
        self.run_guard("""
sys.modules['google.genai'] = object()
try:
    install_guards(forbidden_roots=[], vault_files=[], allow_vault=False, allow_benchmark=False)
except RuntimeError:
    pass
else:
    raise AssertionError('preloaded teacher accepted')
del sys.modules['google.genai']
try:
    install_guards(forbidden_roots=[], vault_files=[], allow_vault='yes', allow_benchmark=False)
except ValueError:
    pass
else:
    raise AssertionError('nonboolean policy accepted')
install_guards(forbidden_roots=[], vault_files=[], allow_vault=False, allow_benchmark=False)
""")

    def test_importlib_source_import_is_blocked(self):
        self.run_guard("""
import importlib
(r/'openai.py').write_text('raise AssertionError("teacher source executed")')
sys.path.insert(0, str(r))
install_guards(forbidden_roots=[], vault_files=[], allow_vault=False, allow_benchmark=False)
blocked(lambda: importlib.import_module('openai'), 'teacher import blocked')
assert 'openai' not in sys.modules
""")

    def test_trusted_schema_exception_blocks_client_constructors_and_requests(self):
        # Authored SDK-layout fixture; actual pinned SDK gets a separate offline check.
        self.run_guard("""
from types import SimpleNamespace
from unittest.mock import patch
from fine_tuning.notebook_rl import guards
class Sync:
    def __init__(self, **kwargs): raise AssertionError('base constructor ran')
    def request(self, *args): raise AssertionError('network request ran')
class Async(Sync): pass
classes = {name: type(name, (Sync,), {'__init__': lambda self, **kw: (_ for _ in ()).throw(
    AssertionError('provider setup ran'))}) for name in
    ('OpenAI', 'AsyncOpenAI', 'AzureOpenAI', 'AsyncAzureOpenAI', 'BedrockOpenAI', 'AsyncBedrockOpenAI')}
sdk = SimpleNamespace(__version__='3.16.2', _client=None, **classes)
base = SimpleNamespace(SyncAPIClient=Sync, AsyncAPIClient=Async)
def dependency(name):
    value = sdk if name == 'openai' else base
    sys.modules[name] = value
    return value
with patch.object(guards, 'version', side_effect=lambda name: '0.11.0' if name == 'vllm' else '3.16.2'), \
     patch.object(guards, 'import_module', side_effect=dependency):
    proof = guards.prepare_vllm_schema_dependency()
assert proof['client_construction_and_requests_blocked'] is True
assert proof['os_network_sandbox'] is False
install_guards(forbidden_roots=[], vault_files=[r/'private'], allow_vault=False, allow_benchmark=False)
for cls in [*classes.values(), Sync, Async]:
    blocked(lambda: cls(api_key='fixture'), 'API client use blocked')
for cls in (Sync, Async):
    blocked(lambda: cls.request(object()), 'API client use blocked')
blocked(lambda: __import__('anthropic'), 'teacher import blocked')
blocked(lambda: (r/'private').read_bytes(), 'vault path blocked')
assert_no_teacher_imports()
Sync.request = lambda *args: None
try: assert_no_teacher_imports()
except RuntimeError: pass
else: raise AssertionError('changed client protection accepted')
""")

    def test_schema_bootstrap_rejects_preloaded_sdk_and_unreviewed_version(self):
        self.run_guard("""
from unittest.mock import patch
from fine_tuning.notebook_rl import guards
sys.modules['openai.types'] = object()
try: guards.prepare_vllm_schema_dependency()
except RuntimeError: pass
else: raise AssertionError('preloaded SDK accepted')
del sys.modules['openai.types']
with patch.object(guards, 'version', return_value='unknown'), patch.object(guards, 'import_module') as imp:
    try: guards.prepare_vllm_schema_dependency()
    except RuntimeError: pass
    else: raise AssertionError('unreviewed version accepted')
    imp.assert_not_called()
""")

    def test_case_insensitive_mount_alias_if_available(self):
        workspace = Path(__file__).resolve().parents[2]
        if not str(workspace).startswith("/mnt/c/"):
            self.skipTest("requires the project Windows mount")
        self.run_guard("""
alias = r/'BENCHMARK'/'data'
if alias.exists():
    install_guards(forbidden_roots=[r/'benchmark'], vault_files=[],
                   allow_vault=False, allow_benchmark=False)
    blocked(lambda: alias.read_bytes(), 'benchmark path blocked')
""", directory=workspace)


if __name__ == "__main__":
    unittest.main()
