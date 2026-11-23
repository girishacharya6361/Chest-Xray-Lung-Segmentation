"""
Download the official Montgomery County Chest X-ray dataset from the
U.S. National Library of Medicine (NLM/LHNCBC).

Usage:
    python download_dataset.py

Files are saved under data/raw/MontgomerySet/. Then prepare the project
dataset with:
    python -m utils.prepare_dataset --src data/raw/MontgomerySet

This script downloads source data; it does not create or fabricate images/masks.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

BASE = (
    "https://data.lhncbc.nlm.nih.gov/public/"
    "Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/"
    "MontgomerySet/"
)
DEST = Path("data/raw/MontgomerySet")
ALLOWED_FILES = {".png", ".txt", ".pdf", ".csv"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self.hrefs.append(value)


def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "ChestXrayAcademicProject/1.0"})
    with urlopen(req, timeout=60) as response:
        return response.read()


def crawl(url: str, relative: Path, visited: set[str]) -> None:
    if url in visited:
        return
    visited.add(url)
    try:
        body = fetch(url)
    except Exception as exc:
        print(f"[WARN] Could not access {url}: {exc}")
        return

    parser = Links()
    try:
        parser.feed(body.decode("utf-8", errors="ignore"))
    except Exception:
        return

    for href in parser.hrefs:
        if href.startswith(("?", "#", "../")):
            continue
        child_url = urljoin(url, href)
        parsed = urlparse(child_url)
        if parsed.netloc != urlparse(BASE).netloc:
            continue
        if not child_url.startswith(BASE):
            continue
        name = href.rstrip("/").split("/")[-1]
        if not name or name in {".", ".."} or name.startswith("index.html"):
            continue

        if href.endswith("/"):
            crawl(child_url, relative / name, visited)
        else:
            suffix = Path(name).suffix.lower()
            if suffix not in ALLOWED_FILES:
                continue
            target = DEST / relative / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > 0:
                continue
            try:
                print(f"Downloading {target}")
                target.write_bytes(fetch(child_url))
            except Exception as exc:
                print(f"[WARN] Failed {child_url}: {exc}")


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    crawl(BASE, Path(), set())
    image_dir = DEST / "CXR_png"
    count = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0
    print(f"\nDownloaded {count} chest X-ray image files.")
    print(f"Source folder: {DEST}")
    if count == 0:
        print("No images were downloaded. Check your internet connection or use the NLM source page.")
    else:
        print("\nNext command:")
        print("python -m utils.prepare_dataset --src data/raw/MontgomerySet")
