"""
s3_utils.py — read public S3 buckets over HTTPS (standard library only).

Used by the motion-energy table scripts (``build_me_table.py``, ``build_me_asset_map.py``,
``check_leading_lost_frames.py``) to read ``aind-open-data`` and ``aind-scratch-data`` without
credentials. Transient errors are retried; a missing object (403 / 404) is ``None``.

Contents
--------
:func:`urlopen`, :func:`download`, :func:`fetch_cached`, :func:`read_json`, :func:`list_keys`
"""

from __future__ import annotations

import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def urlopen(url, tries=4):
    """Open ``url``, retrying transient errors; None on 403 / 404."""
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)


def download(url, dest):
    """Save ``url`` to ``dest``; return ``dest``, or None if it does not exist."""
    response = urlopen(url)
    if response is None:
        return None
    with response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)
    return dest


def fetch_cached(url, cache):
    """Download ``url`` once into the folder ``cache``; return the local path.

    The file name is the object key with ``/`` replaced by ``__``.

    Raises
    ------
    FileNotFoundError
        If the object does not exist.
    """
    path = Path(cache) / url.split("amazonaws.com/", 1)[1].replace("/", "__")
    if not path.exists() and download(url, path) is None:
        raise FileNotFoundError(url)
    return path


def read_json(url):
    """Read a JSON object from ``url``; None if it does not exist."""
    response = urlopen(url)
    if response is None:
        return None
    with response:
        return json.load(response)


def list_keys(bucket_url, prefix):
    """Every object key under ``prefix`` in a bucket, following continuation pages.

    Raises
    ------
    FileNotFoundError
        If the bucket cannot be listed.
    """
    keys, token = [], None
    while True:
        params = {"list-type": 2, "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = urlopen(f"{bucket_url}/?{urllib.parse.urlencode(params)}")
        if response is None:
            raise FileNotFoundError(f"Cannot list {bucket_url}/{prefix}")
        with response:
            text = response.read().decode()
        keys += re.findall(r"<Key>(.*?)</Key>", text)
        token_match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", text)
        if not token_match:
            return keys
        token = token_match.group(1)
