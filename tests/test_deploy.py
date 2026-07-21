#!/usr/bin/env python3
"""Tests for runner/deploy.py — Docker build, push, verify automation."""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# get_next_version
# ---------------------------------------------------------------------------

SAMPLE_CONTEXT = """\
| `v0-submitted` | Jul 9 ~13:43 CDT | Built ~14:15, pushed ~14:20 | Re-saved ~14:20 | **Jul 9, 20:54** | **52.6%** FAILED |
| `v1-no-fireworks` | Jul 9 16:38 CDT | Built ~16:38, pushed ~16:40 | Re-saved ~16:40 | **Never checked** | Skipped |
| `v2-fireworks-030` | Jul 9 17:21 CDT | Built ~17:21, pushed ~17:25 | Re-saved ~17:50 (as `:latest`) | Never checked | Skipped |
| `v5-threshold-010` | Jul 10 02:50 CDT | Built ~02:52, pushed ~02:55 | **Re-save as `:v5`** | Pending | Lowered Fireworks threshold |
"""

CONTEXT_NO_VERSIONS = """\
# Just a header

Some text with no version numbers.
"""

CONTEXT_EMPTY = ""


class TestGetNextVersion:
    def test_finds_max_and_increments(self):
        """Should find v5 and return v6."""
        with patch("runner.deploy.open", mock_open(read_data=SAMPLE_CONTEXT)):
            from runner.deploy import get_next_version
            assert get_next_version() == "v6"

    def test_no_versions_returns_v1(self):
        """When no v<N> found, return v1."""
        with patch("runner.deploy.open", mock_open(read_data=CONTEXT_NO_VERSIONS)):
            from runner.deploy import get_next_version
            assert get_next_version() == "v1"

    def test_empty_file_returns_v1(self):
        """When file is empty, return v1."""
        with patch("runner.deploy.open", mock_open(read_data=CONTEXT_EMPTY)):
            from runner.deploy import get_next_version
            assert get_next_version() == "v1"

    def test_skips_non_version_numbers(self):
        """v followed by non-digit suffix should be ignored."""
        text = "| `vX-test` | something |\n| `v3-real` | something |\n"
        with patch("runner.deploy.open", mock_open(read_data=text)):
            from runner.deploy import get_next_version
            assert get_next_version() == "v4"

    def test_handles_double_digit_version(self):
        """Should handle v10 correctly."""
        text = "| `v9` | data |\n| `v10` | data |\n"
        with patch("runner.deploy.open", mock_open(read_data=text)):
            from runner.deploy import get_next_version
            assert get_next_version() == "v11"


# ---------------------------------------------------------------------------
# pre_build_checks
# ---------------------------------------------------------------------------

class TestPreBuildChecks:
    def test_all_clean_returns_empty(self):
        """When everything is clean, returns empty list."""
        with patch("runner.deploy.subprocess.run") as mock_run, \
             patch("runner.deploy.os.path.exists") as mock_exists, \
             patch("runner.deploy.os.path.islink") as mock_islink:

            mock_run.return_value = MagicMock(returncode=0, stdout=b"")
            mock_exists.return_value = True
            mock_islink.return_value = False

            from runner.deploy import pre_build_checks
            issues = pre_build_checks()
            assert issues == []

    def test_dirty_git_detected(self):
        """Non-empty git status is reported."""
        with patch("runner.deploy.subprocess.run") as mock_run, \
             patch("runner.deploy.os.path.exists") as mock_exists, \
             patch("runner.deploy.os.path.islink") as mock_islink:

            mock_run.return_value = MagicMock(
                returncode=0, stdout=b" M somefile.py\n"
            )
            mock_exists.return_value = True
            mock_islink.return_value = False

            from runner.deploy import pre_build_checks
            issues = pre_build_checks()
            assert any("dirty" in i.lower() or "git" in i.lower() for i in issues)

    def test_missing_dockerfile_reported(self):
        """Missing Dockerfile is reported."""
        with patch("runner.deploy.subprocess.run") as mock_run, \
             patch("runner.deploy.os.path.exists") as mock_exists, \
             patch("runner.deploy.os.path.islink") as mock_islink:

            mock_run.return_value = MagicMock(returncode=0, stdout=b"")

            def exists_side_effect(p):
                if "Dockerfile" in p:
                    return False
                return True

            mock_exists.side_effect = exists_side_effect
            mock_islink.return_value = False

            from runner.deploy import pre_build_checks
            issues = pre_build_checks()
            assert any("Dockerfile" in i for i in issues)

    def test_missing_harness_py_reported(self):
        """Missing harness.py is reported."""
        with patch("runner.deploy.subprocess.run") as mock_run, \
             patch("runner.deploy.os.path.exists") as mock_exists, \
             patch("runner.deploy.os.path.islink") as mock_islink:

            mock_run.return_value = MagicMock(returncode=0, stdout=b"")

            def exists_side_effect(p):
                if "harness.py" in p:
                    return False
                return True

            mock_exists.side_effect = exists_side_effect
            mock_islink.return_value = False

            from runner.deploy import pre_build_checks
            issues = pre_build_checks()
            assert any("harness" in i for i in issues)

    def test_missing_agent_init_reported(self):
        """Missing agent/__init__.py is reported."""
        with patch("runner.deploy.subprocess.run") as mock_run, \
             patch("runner.deploy.os.path.exists") as mock_exists, \
             patch("runner.deploy.os.path.islink") as mock_islink:

            mock_run.return_value = MagicMock(returncode=0, stdout=b"")

            def exists_side_effect(p):
                if "__init__.py" in p and "agent" in p:
                    return False
                return True

            mock_exists.side_effect = exists_side_effect
            mock_islink.return_value = False

            from runner.deploy import pre_build_checks
            issues = pre_build_checks()
            assert any("agent/__init__" in i or "agent" in i for i in issues)

    def test_broken_model_symlink_reported(self):
        """A broken symlink for the model is flagged."""
        with patch("runner.deploy.subprocess.run") as mock_run, \
             patch("runner.deploy.os.path.exists") as mock_exists, \
             patch("runner.deploy.os.path.islink") as mock_islink:

            mock_run.return_value = MagicMock(returncode=0, stdout=b"")
            mock_exists.side_effect = lambda p: False if "models/" in p else True
            mock_islink.return_value = True

            from runner.deploy import pre_build_checks
            issues = pre_build_checks()
            assert any("model" in i.lower() or "broken" in i.lower()
                       for i in issues)


# ---------------------------------------------------------------------------
# build_image
# ---------------------------------------------------------------------------

class TestBuildImage:
    def test_build_success(self):
        """Successful build returns True."""
        with patch("runner.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            from runner.deploy import build_image
            result = build_image("v6")
            assert result is True
            # Check the docker command was constructed
            cmd = mock_run.call_args[0][0]
            assert "docker" in cmd
            assert "buildx" in cmd

    def test_build_failure(self):
        """Failed build returns False."""
        with patch("runner.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            from runner.deploy import build_image
            result = build_image("v6")
            assert result is False

    def test_platform_flag_passed(self):
        """Custom platform is passed to docker buildx."""
        with patch("runner.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            from runner.deploy import build_image
            build_image("v7", platform="linux/arm64")
            cmd = " ".join(mock_run.call_args[0][0])
            assert "linux/arm64" in cmd

    def test_no_cache_flag(self):
        """--no-cache flag is passed when no_cache=True."""
        with patch("runner.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            from runner.deploy import build_image
            build_image("v8", no_cache=True)
            cmd = " ".join(mock_run.call_args[0][0])
            assert "--no-cache" in cmd


# ---------------------------------------------------------------------------
# push_image
# ---------------------------------------------------------------------------

class TestPushImage:
    def test_push_success(self):
        """Successful push returns True."""
        with patch("runner.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            from runner.deploy import push_image
            result = push_image("v6")
            assert result is True
            cmd = mock_run.call_args[0][0]
            assert "push" in cmd

    def test_push_failure(self):
        """Failed push returns False."""
        with patch("runner.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            from runner.deploy import push_image
            result = push_image("v6")
            assert result is False


# ---------------------------------------------------------------------------
# verify_image
# ---------------------------------------------------------------------------

class TestVerifyImage:
    def test_returns_dict_with_expected_keys(self):
        """verify_image returns dict with check names and passed/detail."""
        with patch("runner.deploy.subprocess.run") as mock_run:
            # All docker commands succeed
            mock_run.return_value = MagicMock(
                returncode=0, stdout=b"linux/amd64"
            )

            from runner.deploy import verify_image
            result = verify_image("v6")

            assert isinstance(result, dict)
            assert "exists" in result
            assert "platform" in result
            assert "entrypoint" in result
            assert "imports" in result
            # Check result shape: each entry has passed and detail
            for key in ("exists", "platform", "entrypoint", "imports"):
                assert "passed" in result[key]
                assert "detail" in result[key]

    def test_image_not_found(self):
        """When image doesn't exist, mark checks as failed."""
        with patch("runner.deploy.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            from runner.deploy import verify_image
            result = verify_image("nonexistent")
            assert result["exists"]["passed"] is False


# ---------------------------------------------------------------------------
# deploy() integration flow
# ---------------------------------------------------------------------------

class TestDeploy:
    def test_deploy_dry_run_no_tag_generates_version(self):
        """deploy() without a tag auto-generates one via get_next_version."""
        with patch("runner.deploy.get_next_version", return_value="v99"), \
             patch("runner.deploy.pre_build_checks", return_value=[]), \
             patch("runner.deploy.build_image", return_value=True), \
             patch("runner.deploy.verify_image") as mock_verify, \
             patch("runner.deploy.update_context_md"):

            mock_verify.return_value = {
                "exists": {"passed": True, "detail": "ok"},
                "platform": {"passed": True, "detail": "linux/amd64"},
                "entrypoint": {"passed": True, "detail": "ok"},
                "imports": {"passed": True, "detail": "ok"},
            }

            from runner.deploy import deploy
            rc = deploy(verify=True, push=False, update_context=False)
            assert rc == 0

    def test_deploy_pre_check_failure_aborts(self):
        """Pre-build failures should abort before building."""
        with patch("runner.deploy.get_next_version", return_value="v99"), \
             patch("runner.deploy.pre_build_checks",
                   return_value=["Dirty git", "Missing Dockerfile"]), \
             patch("runner.deploy.build_image") as mock_build:

            from runner.deploy import deploy
            rc = deploy(verify=False, push=False, update_context=False)
            assert rc == 1
            mock_build.assert_not_called()

    def test_deploy_build_failure_aborts(self):
        """Build failure should abort and return 1."""
        with patch("runner.deploy.get_next_version", return_value="v99"), \
             patch("runner.deploy.pre_build_checks", return_value=[]), \
             patch("runner.deploy.build_image", return_value=False), \
             patch("runner.deploy.push_image") as mock_push:

            from runner.deploy import deploy
            rc = deploy(verify=False, push=False, update_context=False)
            assert rc == 1
            mock_push.assert_not_called()

    def test_deploy_with_push_calls_push(self):
        """When push=True, push_image is called."""
        with patch("runner.deploy.get_next_version", return_value="v99"), \
             patch("runner.deploy.pre_build_checks", return_value=[]), \
             patch("runner.deploy.build_image", return_value=True), \
             patch("runner.deploy.push_image", return_value=True) as mock_push, \
             patch("runner.deploy.verify_image") as mock_verify, \
             patch("runner.deploy.update_context_md"):

            mock_verify.return_value = {
                "exists": {"passed": True, "detail": "ok"},
                "platform": {"passed": True, "detail": "linux/amd64"},
                "entrypoint": {"passed": True, "detail": "ok"},
                "imports": {"passed": True, "detail": "ok"},
            }

            from runner.deploy import deploy
            rc = deploy(push=True, verify=True, update_context=False)
            assert rc == 0
            mock_push.assert_called_once()


# ---------------------------------------------------------------------------
# CLI / __main__ parsing
# ---------------------------------------------------------------------------

class TestCLI:
    def test_parse_args_build_only(self):
        """--build-only should set build_only=True and verify=False."""
        from runner.deploy import parse_args
        args = parse_args(["--build-only"])
        assert args.build_only is True
        assert args.push is False
        assert args.verify is False

    def test_parse_args_push(self):
        """--push should set push=True."""
        from runner.deploy import parse_args
        args = parse_args(["--push"])
        assert args.push is True

    def test_parse_args_verify(self):
        """--verify should set verify=True."""
        from runner.deploy import parse_args
        args = parse_args(["--verify"])
        assert args.verify is True

    def test_parse_args_tag(self):
        """--tag should set tag value."""
        from runner.deploy import parse_args
        args = parse_args(["--tag", "v42"])
        assert args.tag == "v42"

    def test_parse_args_no_update_context(self):
        """--no-update-context should set update_context=False."""
        from runner.deploy import parse_args
        args = parse_args(["--push", "--no-update-context"])
        assert args.update_context is False

    def test_parse_args_build_only_implies_no_push_no_verify(self):
        """--build-only implies verify=False even when --verify is not given."""
        from runner.deploy import parse_args
        args = parse_args(["--build-only"])
        assert args.push is False
        assert args.verify is False

    def test_main_invokes_deploy(self):
        """__main__ block should call deploy with parsed args."""
        with patch("runner.deploy.deploy", return_value=0) as mock_deploy:
            from runner.deploy import main
            rc = main(["--build-only"])
            assert rc == 0
            mock_deploy.assert_called_once()
            kwargs = mock_deploy.call_args[1]
            assert kwargs.get("push") is False
            assert kwargs.get("verify") is False
