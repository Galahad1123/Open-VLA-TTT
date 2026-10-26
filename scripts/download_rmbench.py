#!/usr/bin/env python3
"""Download pinned RMBench resources; large payloads always bypass proxies."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from vla_ttt_eval.rmbench import DATA_REVISION, resolve_tasks

REPO = "TianxingChen/RMBench"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def download(relative, target, size, sha, endpoint):
    """Resume bytes directly; metadata HEAD may use the workstation's proxy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.stat().st_size == size and (sha is None or digest == sha):
            return
        raise ValueError(f"Existing file differs: {target}")
    partial = target.with_suffix(target.suffix + ".partial")
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"{endpoint}/datasets/{REPO}/resolve/{DATA_REVISION}/{relative}?download=true"
    # The pinned repo stores small files (at most 2 MiB) directly in Git.
    # Fall back to the official small-file endpoint when the mirror is unreliable.
    if sha is None and size <= 2 * 1024 * 1024:
        small_url = url.replace("https://hf-mirror.com/", "https://huggingface.co/")
        with urllib.request.urlopen(small_url, timeout=60) as response:
            content = response.read(size + 1)
        if len(content) != size:
            raise ValueError(f"Size mismatch: {relative}")
        partial.write_bytes(content)
        partial.rename(target)
        return
    if endpoint == "https://hf-mirror.com":
        # This mirror accepts curl where urllib may time out or receive HTTP 403.
        subprocess.run(
            [
                "curl",
                "--noproxy",
                "*",
                "-4",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "15",
                "--max-time",
                "600",
                "--retry",
                "2",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        with partial.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if partial.stat().st_size != size or (sha and digest != sha):
            raise ValueError(f"Size/checksum mismatch: {partial}")
        partial.rename(target)
        return
    # Resolve metadata only through proxy if needed. Never GET a payload with this opener.
    try:
        response = urllib.request.build_opener(NoRedirect()).open(
            urllib.request.Request(url, method="HEAD"), timeout=30
        )
        response.close()
        location = url
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308):
            raise
        location = urllib.parse.urljoin(url, error.headers["Location"])
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        location, headers={"Range": f"bytes={offset}-"} if offset else {}
    )
    if offset < size:
        with direct.open(request, timeout=60) as response:
            if offset and response.status != 206:
                offset = 0
            with partial.open("ab" if offset else "wb") as stream:
                while chunk := response.read(4 * 1024 * 1024):
                    stream.write(chunk)
    with partial.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if partial.stat().st_size != size or (sha and digest != sha):
        raise ValueError(f"Size/checksum mismatch: {partial}")
    partial.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--kind", choices=["data", "assets"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--metadata-endpoint", help="Separate small API metadata endpoint")
    parser.add_argument("--manifest", type=Path, help="Reuse a pinned JSON resource listing")
    parser.add_argument("--download", action="store_true", help="Without this flag only list sizes")
    args = parser.parse_args()
    tasks = resolve_tasks(args.tasks)
    endpoint = args.endpoint.rstrip("/")
    files = []
    metadata_endpoint = (args.metadata_endpoint or endpoint).rstrip("/")
    url = f"{metadata_endpoint}/api/datasets/{REPO}/tree/{DATA_REVISION}?recursive=true&limit=1000"
    # Metadata only; urllib also tolerates workstations with an ALL_PROXY socks alias.
    if args.manifest:
        cached = json.loads(args.manifest.read_text())
        if cached["revision"] != DATA_REVISION:
            raise ValueError("Cached resource revision differs")
        files = [dict(row, type="file", lfs={"oid": row["sha256"]}) for row in cached["files"]]
        url = None
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            files.extend(json.load(response))
            next_page = re.search(r'<([^>]+)>; rel="next"', response.headers.get("Link", ""))
            url = next_page.group(1) if next_page else None
    wanted = {f"data/{task}/demo_clean.zip" for task in tasks}
    entries = []
    for item in files:
        if item["type"] != "file":
            continue
        if (args.kind == "data" and item["path"] in wanted) or (
            args.kind == "assets" and item["path"].startswith(("embodiments/", "objects/"))
        ):
            entries.append(
                {
                    "path": item["path"],
                    "size": item["size"],
                    "sha256": item.get("lfs", {}).get("oid"),
                }
            )
    if not entries or (args.kind == "data" and len(entries) != len(tasks)):
        raise ValueError("Resource manifest is incomplete")
    print(
        json.dumps(
            {
                "revision": DATA_REVISION,
                "files": entries,
                "total_GiB": sum(x["size"] for x in entries) / 2**30,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.download:

        def transfer(item):
            for attempt in range(4):
                try:
                    print(f"Preparing resource: {item['path']}", flush=True)
                    download(
                        item["path"],
                        args.output / item["path"],
                        item["size"],
                        item["sha256"],
                        endpoint,
                    )
                    return None
                except (OSError, urllib.error.URLError, subprocess.CalledProcessError) as error:
                    if attempt == 3:
                        return f"{item['path']}: {error}"
                    time.sleep(2**attempt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            errors = [error for error in pool.map(transfer, entries) if error]
        if errors:
            raise RuntimeError("Downloads incomplete; rerun to resume:\n" + "\n".join(errors))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{args.kind}-download-manifest.json").write_text(
            json.dumps(entries, indent=2)
        )


if __name__ == "__main__":
    main()
