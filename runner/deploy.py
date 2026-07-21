#!/usr/bin/env python3
"""Docker build, push, verify automation for AMD ACT II Hackathon.

Usage:
    python -m runner.deploy --build-only           # build only
    python -m runner.deploy --push --verify         # build + push + verify
    python -m runner.deploy --tag v42 --push        # tag-specific deploy
    python -m runner.deploy --no-cache --push       # fresh build + push
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = "ghcr.io/artemkorolev1/amd-hackathon-submit"
DOCKERFILE = "/home/artem/dev/amd-hackathon/Dockerfile"
BUILD_CTX = "/home/artem/dev/amd-hackathon"
CONTEXT_MD = os.path.join(BUILD_CTX, "CONTEXT.md")
MODELS_DIR = os.path.join(BUILD_CTX, "models")


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def get_next_version() -> str:
    """Read CONTEXT.md, find the highest v<N> tag, return v<N+1>.

    Scans for patterns like `` `v<N>-...` `` or `` `v<N>` `` in the file,
    extracts the numeric part, increments by one.
    If no version is found, returns "v1".
    """
    if not os.path.exists(CONTEXT_MD):
        return "v1"

    with open(CONTEXT_MD, "r") as f:
        content = f.read()

    # Match backtick-wrapped v<N> patterns (e.g. `v5-threshold-010`, `v3`)
    versions = re.findall(r"`v(\d+)", content)
    if not versions:
        return "v1"

    max_ver = max(int(v) for v in versions)
    return f"v{max_ver + 1}"


# ---------------------------------------------------------------------------
# Pre-build checks
# ---------------------------------------------------------------------------

def pre_build_checks() -> list[str]:
    """Run pre-flight checks before building.

    Returns a list of issue descriptions.  Empty list means all clear.
    """
    issues: list[str] = []

    # 1. Git working tree clean
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            cwd=BUILD_CTX,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            issues.append(
                f"Git working tree is dirty:\n{result.stdout.decode()}"
            )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        issues.append(f"Could not check git status: {exc}")

    # 2. Required files exist
    required_files = [
        ("Dockerfile", DOCKERFILE),
        ("harness.py", os.path.join(BUILD_CTX, "harness.py")),
        ("agent/__init__.py", os.path.join(BUILD_CTX, "agent", "__init__.py")),
    ]
    for name, path in required_files:
        if not os.path.exists(path):
            issues.append(f"Missing {name} — expected at {path}")

    # 3. Model file is real (not a symlink)
    if os.path.isdir(MODELS_DIR):
        for entry in os.listdir(MODELS_DIR):
            if entry.endswith(".gguf"):
                model_path = os.path.join(MODELS_DIR, entry)
                if os.path.islink(model_path):
                    issues.append(
                        f"Model is a symlink, must be real file: {model_path}"
                    )
                elif not os.path.isfile(model_path):
                    issues.append(f"Model file missing: {model_path}")

    return issues


# ---------------------------------------------------------------------------
# Docker operations
# ---------------------------------------------------------------------------

def build_image(
    tag: str, platform: str = "linux/amd64", no_cache: bool = False
) -> bool:
    """Run ``docker buildx build`` and return True on success."""
    image_ref = f"{DEFAULT_IMAGE}:{tag}"
    cmd = [
        "docker", "buildx", "build",
        "--platform", platform,
        "-t", image_ref,
        "--load",
    ]
    if no_cache:
        cmd.append("--no-cache")
    cmd.append(BUILD_CTX)

    print(f"  Building {image_ref} ...")
    print(f"  Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            print(f"  BUILD FAILED:\n{result.stderr.decode()}")
            return False
        print("  Build succeeded.")
        return True
    except subprocess.TimeoutExpired:
        print("  Build timed out after 600s.")
        return False
    except FileNotFoundError:
        print("  ERROR: docker not found on PATH.")
        return False


def push_image(tag: str) -> bool:
    """Run ``docker push`` and return True on success."""
    image_ref = f"{DEFAULT_IMAGE}:{tag}"
    cmd = ["docker", "push", image_ref]

    print(f"  Pushing {image_ref} ...")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            print(f"  PUSH FAILED:\n{result.stderr.decode()}")
            return False
        print("  Push succeeded.")
        return True
    except subprocess.TimeoutExpired:
        print("  Push timed out after 300s.")
        return False
    except FileNotFoundError:
        print("  ERROR: docker not found on PATH.")
        return False


def verify_image(tag: str) -> dict:
    """Run verification checks on a built image.

    Returns a dict mapping check names to ``{"passed": bool, "detail": str}``.
    """
    image_ref = f"{DEFAULT_IMAGE}:{tag}"
    results: dict[str, dict] = {}

    # 1. Image exists locally
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_ref],
            capture_output=True,
            timeout=30,
        )
        exists = result.returncode == 0
        results["exists"] = {
            "passed": exists,
            "detail": "Image found" if exists else "Image not found locally",
        }
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        results["exists"] = {
            "passed": False,
            "detail": f"Inspect error: {exc}",
        }
        # Can't continue without the image
        for check in ("platform", "entrypoint", "imports"):
            results[check] = {"passed": False, "detail": "Skipped — image not found"}
        return results

    if not results["exists"]["passed"]:
        for check in ("platform", "entrypoint", "imports"):
            results[check] = {"passed": False, "detail": "Skipped — image not found"}
        return results

    # 2. Platform check
    try:
        result = subprocess.run(
            [
                "docker", "inspect", image_ref,
                "--format", "{{.Os}}/{{.Architecture}}",
            ],
            capture_output=True,
            timeout=15,
        )
        plat = result.stdout.decode().strip()
        passed = plat == "linux/amd64"
        results["platform"] = {
            "passed": passed,
            "detail": plat if passed else f"Expected linux/amd64, got {plat}",
        }
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        results["platform"] = {"passed": False, "detail": str(exc)}

    # 3. Entrypoint check
    try:
        result = subprocess.run(
            [
                "docker", "inspect", image_ref,
                "--format", "{{json .Config.Entrypoint}}",
            ],
            capture_output=True,
            timeout=15,
        )
        entrypoint = result.stdout.decode().strip()
        passed = "python3" in entrypoint and "harness.py" in entrypoint
        results["entrypoint"] = {
            "passed": passed,
            "detail": entrypoint if not passed else "Entrypoint OK",
        }
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        results["entrypoint"] = {"passed": False, "detail": str(exc)}

    # 4. Import check (quick validation that agent module loads)
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--entrypoint", "python3",
                image_ref,
                "-c", "from agent import Pipeline; print('OK')",
            ],
            capture_output=True,
            timeout=30,
        )
        passed = result.returncode == 0 and b"OK" in result.stdout
        results["imports"] = {
            "passed": passed,
            "detail": "Import OK" if passed else result.stderr.decode().strip()[:200],
        }
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        results["imports"] = {"passed": False, "detail": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Context.md updater
# ---------------------------------------------------------------------------

def update_context_md(
    tag: str,
    build_ok: bool,
    push_ok: bool,
    verify_results: dict,
) -> None:
    """Append a version entry to CONTEXT.md."""
    from datetime import datetime

    now = datetime.now().strftime("%b %d %H:%M CDT")
    build_status = f"Built ~{now}" if build_ok else "BUILD FAILED"
    push_status = f"pushed ~{now}" if push_ok else "not pushed"
    verify_detail = ""
    if verify_results:
        all_pass = all(
            v.get("passed", False) for v in verify_results.values()
        )
        verify_detail = "✅ All checks passed" if all_pass else "⚠️ Some checks failed"

    entry = (
        f"| `{tag}` | {now} | {build_status}, {push_status} |"
        f" — | — | {verify_detail} |\n"
    )

    with open(CONTEXT_MD, "a") as f:
        f.write(entry)

    print(f"  Updated CONTEXT.md with entry for {tag}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def deploy(
    tag: str | None = None,
    push: bool = False,
    verify: bool = True,
    update_context: bool = True,
    build_only: bool = False,
    platform: str = "linux/amd64",
    no_cache: bool = False,
) -> int:
    """Full deploy pipeline: pre-check → build → (optional push) → verify → document.

    Returns 0 on success, 1 on failure.
    """
    if tag is None:
        tag = get_next_version()
    print(f"=== Deploy v{tag} ===")

    # 1. Pre-build checks
    print("\n--- Pre-build checks ---")
    issues = pre_build_checks()
    if issues:
        print("ISSUES FOUND:")
        for i in issues:
            print(f"  • {i}")
        print("Aborting.")
        return 1
    print("  All checks passed.")

    # 2. Build
    print("\n--- Build ---")
    build_ok = build_image(tag, platform=platform, no_cache=no_cache)
    if not build_ok:
        print("Build failed. Aborting.")
        if update_context:
            update_context_md(tag, build_ok=False, push_ok=False, verify_results={})
        return 1

    if build_only:
        print(f"\nBuild-only mode. Image tagged as {DEFAULT_IMAGE}:{tag}")
        if update_context:
            update_context_md(tag, build_ok=True, push_ok=False, verify_results={})
        return 0

    # 3. Push (optional)
    push_ok = False
    if push:
        print("\n--- Push ---")
        push_ok = push_image(tag)
        if not push_ok:
            print("Push failed.")
            if update_context:
                update_context_md(
                    tag, build_ok=True, push_ok=False, verify_results={}
                )
            return 1
    else:
        print("\n--- Push skipped (use --push to enable) ---")

    # 4. Verify (optional)
    verify_results: dict = {}
    if verify:
        print("\n--- Verify ---")
        verify_results = verify_image(tag)
        all_pass = all(
            v.get("passed", False) for v in verify_results.values()
        )
        for check_name, result in verify_results.items():
            icon = "✅" if result.get("passed") else "❌"
            print(f"  {icon} {check_name}: {result.get('detail', '')}")
        if not all_pass:
            print("WARNING: Some verification checks failed (continuing anyway).")
    else:
        print("\n--- Verify skipped (use --verify to enable) ---")

    # 5. Update CONTEXT.md
    if update_context:
        update_context_md(tag, build_ok, push_ok, verify_results)

    print(f"\n=== Deploy complete: {DEFAULT_IMAGE}:{tag} ===")
    return 0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Docker build/push/verify for AMD ACT II Hackathon",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Image tag (e.g. v42). Auto-incremented from CONTEXT.md if omitted.",
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="Only build the image (skip push, verify, context update).",
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Push image to GHCR after building.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run verification checks on built image.",
    )
    parser.add_argument(
        "--no-update-context", dest="update_context",
        action="store_false", default=True,
        help="Skip updating CONTEXT.md with deploy entry.",
    )
    # Positive flag for explicit use
    parser.add_argument(
        "--update-context", dest="update_context",
        action="store_true",
        help="Update CONTEXT.md with deploy entry (default: yes).",
    )
    parser.add_argument(
        "--platform", type=str, default="linux/amd64",
        help="Target platform (default: linux/amd64).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Pass --no-cache to docker build.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build_only:
        return deploy(
            tag=args.tag,
            push=False,
            verify=False,
            update_context=args.update_context,
            build_only=True,
            platform=args.platform,
            no_cache=args.no_cache,
        )
    return deploy(
        tag=args.tag,
        push=args.push,
        verify=args.verify,
        update_context=args.update_context,
        platform=args.platform,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    sys.exit(main())
