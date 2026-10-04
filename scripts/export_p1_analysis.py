"""Repackage preserved P1 as one code ZIP and six standalone text HTMLs.

Only generated image/index links are removed. Raw responses and all scoring
metadata remain unchanged. The full bundle and images stay in the source archive.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from mmdl.runtime.artifacts import atomic_text

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis_exports/mmmu_20260924/p1_full_archive"
DEST = ROOT / "results/P1"
RUNS = {"baseline_2048": "2K", "continuous_32k": "32K", "official_b2": "32K_highres"}
ZIP_NAME = "V6_V7_code.zip"
IMAGE_LINE = re.compile(
    r'<p><a href="\.\./\.\./assets/images/[a-f0-9]{64}\.png">'
    r'<img src="\.\./\.\./assets/images/[a-f0-9]{64}\.png"[^>]*></a></p>\n?'
)
BACK_LINK = '<a href="../../index.html">All six results</a>'
NOTICE = ("<p>웹 업로드용 파싱 분석 결과: 질문·보기·모델 응답 전문·정답·채점 정보를 보존했습니다. "
          "이미지 픽셀은 이 HTML에 포함하지 않습니다. 원본 이미지와 전체 자료는 별도로 보존되어 있습니다.</p>")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def html_text(version, run):
    page = (SOURCE / "runs" / f"{version}_{run}" / "review.html").read_text(encoding="utf-8")
    page, count = IMAGE_LINE.subn("", page)
    require(count == 982 and page.count("<section>") == 900, "Unexpected source HTML layout")
    require(page.count(BACK_LINK) == 1, "Unexpected source navigation")
    page = page.replace(BACK_LINK, "").replace("</h1>", "</h1>" + NOTICE, 1)
    require("<img " not in page and '<a href="' not in page, "Remaining external asset/link")
    return page


def html_name(version, run):
    return f"{version}_{RUNS[run]}.html"


def verify():
    expected = {ZIP_NAME} | {html_name(v, r) for v in ("V6", "V7") for r in RUNS}
    require({p.name for p in DEST.iterdir()} == expected
            and all((DEST / name).is_file() for name in expected), "P1 must contain exactly seven files")
    with zipfile.ZipFile(DEST / ZIP_NAME) as combined:
        expected_members = set()
        require(combined.testzip() is None, "Combined ZIP checksum failure")
        for version in ("V6", "V7"):
            with zipfile.ZipFile(SOURCE / "code" / f"{version}_code.zip") as original:
                for name in original.namelist():
                    member = f"{version}/{name}"
                    expected_members.add(member)
                    require(combined.read(member) == original.read(name), "Source ZIP bytes changed")
        require(set(combined.namelist()) == expected_members
                and len(combined.namelist()) == len(expected_members), "Unexpected or duplicate ZIP member")

    def check(pair):
        version, run = pair
        path = DEST / html_name(version, run)
        require(path.read_text(encoding="utf-8") == html_text(version, run), "HTML text changed unexpectedly")
        return f"{path.name}: 900 records, original text/scoring preserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(check, [(v, r) for v in ("V6", "V7") for r in RUNS]):
            print(result, flush=True)
    print("VERIFIED: exactly 1 code ZIP + 6 standalone text HTMLs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
        return
    require(SOURCE.is_dir(), "Missing preserved full P1 archive")
    require(not DEST.exists() or not any(DEST.iterdir()), "Refusing to overwrite an existing P1 export")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DEST / ZIP_NAME, "x", zipfile.ZIP_DEFLATED) as combined:
        for version in ("V6", "V7"):
            with zipfile.ZipFile(SOURCE / "code" / f"{version}_code.zip") as original:
                for info in original.infolist():
                    name = PurePosixPath(info.filename)
                    require(not name.is_absolute() and ".." not in name.parts, "Unsafe source ZIP path")
                    combined.writestr(f"{version}/{info.filename}", original.read(info))

    def write(pair):
        version, run = pair
        atomic_text(DEST / html_name(version, run), html_text(version, run))
        print(f"Created {html_name(version, run)}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, [(v, r) for v in ("V6", "V7") for r in RUNS]))
    verify()


if __name__ == "__main__":
    main()
