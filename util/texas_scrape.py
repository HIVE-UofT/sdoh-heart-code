"""
Scrape the Texas Heart Institute Cardiovascular Glossary
and save it as texas_heart_dictionary.csv with two columns:
term, definition
"""
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

URL = "https://www.texasheart.org/heart-health/heart-information-center/topics/a-z/"
OUT_FILE = Path("texas_heart_dictionary.csv")

# One global Session with desktop-browser headers
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
})

DASH_RE = re.compile(r"\s+–\s+")        # en-dash surrounded by spaces

def fetch_glossary(url: str) -> list[dict]:
    resp = SESSION.get(url, timeout=30)
    if resp.status_code == 403:
        raise RuntimeError(
            "403 Forbidden – the site is still blocking us. "
            "Try a different network, slower rate, or a library such as cloudscraper."
        )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []

    for p in soup.select("p"):
        text = " ".join(p.get_text(" ").split())
        if DASH_RE.search(text):
            term, definition = DASH_RE.split(text, maxsplit=1)
            records.append({"term": term.strip(), "definition": definition.strip()})
    return records

def main() -> None:
    items = fetch_glossary(URL)
    if not items:
        raise RuntimeError("No glossary terms parsed – page structure may have changed.")
    pd.DataFrame(items).to_csv(OUT_FILE, index=False)
    print(f"Saved {len(items):,} terms to {OUT_FILE}")

if __name__ == "__main__":
    main()
