"""Download the AI4Arctic ready-to-train dataset from DTU Data (figshare)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ARTICLE_ID = 21316608
DATA_DIR = Path("data")
N_WORKERS = 4  # parallel downloads: figshare limits the speed of each single connection


def list_files():
    url = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()  # stop with an error if the request failed
    return response.json()["files"]


def is_complete(file_info):
    # A partial download exists but is smaller than the expected size
    target = DATA_DIR / file_info["name"]
    return target.exists() and target.stat().st_size == file_info["size"]


def download(file_info):
    target = DATA_DIR / file_info["name"]
    # timeout: give up if the server sends nothing for 60 s, instead of hanging forever
    with requests.get(file_info["download_url"], stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(target, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB at a time
                f.write(chunk)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    files = list_files()
    todo = [f for f in files if not is_complete(f)]
    todo_gb = sum(f["size"] for f in todo) / 1024**3
    print(f"{len(files) - len(todo)} files already downloaded, {len(todo)} to go ({todo_gb:.1f} GB)")

    failed = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(download, f): f["name"] for f in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                future.result()  # re-raises here any error that happened in the download
                print(f"[{done}/{len(todo)}] {name}")
            except Exception as error:
                failed.append(name)
                print(f"[{done}/{len(todo)}] FAILED {name}: {error}")

    if failed:
        print(f"{len(failed)} files failed: run the script again to retry them.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()