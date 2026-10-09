"""Download model assets (the MediaPipe face landmarker) with retries and atomic placement."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from media_culler.exceptions import CullerError

log = logging.getLogger(__name__)
_CHUNK = 64 * 1024
_MIN_PLAUSIBLE_BYTES = 1024


class DownloadError(CullerError):
    pass


def download_file(
    url: str,
    destination: Path,
    *,
    sha256: str | None = None,
    retries: int = 3,
    timeout: float = 30.0,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ai-media-culler"})
            digest = hashlib.sha256()
            received = 0
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                partial.open("wb") as out,
            ):
                length = response.headers.get("Content-Length")
                total = int(length) if length and length.isdigit() else None
                while chunk := response.read(_CHUNK):
                    out.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if on_progress:
                        on_progress(received, total)

            if received < _MIN_PLAUSIBLE_BYTES:
                raise DownloadError(f"Download too small ({received} bytes); wrong URL?")
            if sha256 and digest.hexdigest().lower() != sha256.lower():
                raise DownloadError("SHA-256 mismatch; refusing to install the downloaded file")
            os.replace(partial, destination)
            return destination
        except DownloadError:
            partial.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last_error = error
            log.warning("Download attempt %d/%d failed: %s", attempt, retries, error)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
        finally:
            partial.unlink(missing_ok=True) if partial.exists() else None

    raise DownloadError(f"Could not download {url}: {last_error}")
