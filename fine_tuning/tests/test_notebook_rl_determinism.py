"""CPU controls of native deterministic settings; not CUDA evidence."""
import os
import unittest
from unittest.mock import patch

import torch

from fine_tuning.notebook_rl.policy import enable_deterministic_execution


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warning = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, enabled, warn_only=warning)

    def test_configured_before_cuda_and_strict_not_warning_only(self):
        with patch.dict(os.environ), patch.object(torch.cuda, "is_initialized", return_value=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            torch.use_deterministic_algorithms(False)
            actual = enable_deterministic_execution()
            self.assertEqual(actual, {"algorithms_enabled": True, "warn_only": False,
                                      "cublas_workspace_config": ":4096:8"})
            self.assertEqual(enable_deterministic_execution(), actual)

    def test_late_configuration_and_conflicting_workspace_are_rejected(self):
        with patch.dict(os.environ), patch.object(torch.cuda, "is_initialized", return_value=True):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with self.assertRaisesRegex(ValueError, "before CUDA"):
                enable_deterministic_execution()
            self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            with self.assertRaisesRegex(ValueError, "requires CUBLAS"):
                enable_deterministic_execution()

    def test_unsupported_deterministic_operation_is_not_suppressed(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                patch.object(torch, "use_deterministic_algorithms", side_effect=RuntimeError("unsupported")):
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                enable_deterministic_execution()
