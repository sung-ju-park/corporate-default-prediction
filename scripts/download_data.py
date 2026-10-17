"""UCI 저장소에서 원본 데이터를 받아 data/raw/data.csv 로 저장한다.

    python scripts/download_data.py

자동 다운로드가 안 되면 아래 페이지에서 직접 받아 압축을 풀고,
안의 data.csv 를 data/raw/ 폴더에 넣으면 된다.
https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "data.csv"
EXPECTED_SHAPE = (6819, 96)


def main() -> None:
    if OUT.exists():
        print(f"이미 있습니다: {OUT}")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"내려받는 중: {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            payload = resp.read()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"다운로드 실패: {exc}\n위 주석의 페이지에서 직접 받아 {OUT} 에 넣어 주세요.")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            sys.exit(f"압축 파일 안에 CSV가 없습니다: {zf.namelist()}")
        OUT.write_bytes(zf.read(csvs[0]))

    import pandas as pd
    df = pd.read_csv(OUT)
    if df.shape != EXPECTED_SHAPE:
        print(f"주의: 예상 크기 {EXPECTED_SHAPE}와 다릅니다 -> {df.shape}")
    print(f"저장 완료: {OUT} {df.shape}, 부도 {int(df.iloc[:, 0].sum())}개")


if __name__ == "__main__":
    main()
