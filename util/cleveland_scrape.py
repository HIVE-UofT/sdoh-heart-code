# """
# File to scrape heart terms from website containing cardiovascular dictionary.

# Parse into csv file with 2 columns: term and definition.
# """

# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm
# import numpy as np


# # website to scrape
# CLEVELAND = [
#     "https://my.clevelandclinic.org/departments/heart/patient-education/dictionary#a-c-tab", 
#     "https://my.clevelandclinic.org/departments/heart/patient-education/dictionary#d-f-tab", 
#     "https://my.clevelandclinic.org/departments/heart/patient-education/dictionary#g-i-tab", 
#     "https://my.clevelandclinic.org/departments/heart/patient-education/dictionary#j-o-tab", 
#     "https://my.clevelandclinic.org/departments/heart/patient-education/dictionary#p-s-tab", 
#     "https://my.clevelandclinic.org/departments/heart/patient-education/dictionary#t-z-tab"
# ]


# # explore scraped raw content

# test = CLEVELAND[0]

# from bs4 import BeautifulSoup
# import requests
# response = requests.get(test)
# soup = BeautifulSoup(response.content, 'html.parser')

# # get raw text from website
# raw_text = soup.get_text(separator='\n', strip=True)

# # split into lines
# lines = raw_text.split('\n')
# # 
# print("First 10 lines of raw text:")
# for line in lines[:10]:
#     print(line)


"""
Scrape the Cleveland Clinic Cardiovascular Dictionary and save
it as cardio_dictionary.csv with two columns: term, definition.
"""

import requests
from bs4 import BeautifulSoup, NavigableString
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://my.clevelandclinic.org/departments/heart/patient-education/dictionary"
TAB_IDS = [
    "a-c-tab", "d-f-tab", "g-i-tab",
    "j-o-tab", "p-s-tab", "t-z-tab",
]

def clean_text(s: str) -> str:
    """Collapse whitespace and remove leading/trailing space."""
    return " ".join(s.split())

def extract_pairs(soup: BeautifulSoup, tab_id: str) -> list[tuple[str, str]]:
    """Return (term, definition) pairs from one tab."""
    container = soup.find(id=tab_id)
    if not container:
        return []

    pairs = []
    for p in container.find_all("p", recursive=False):
        strong = p.find("strong")
        if not strong:
            continue                       # skip headers or malformed rows

        # term text (strip any <br/> inside <strong>)
        term = clean_text(strong.get_text(" ", strip=True))

        # definition = everything after the <strong> tag
        definition_parts = []
        for elem in strong.next_siblings:
            if isinstance(elem, NavigableString):
                definition_parts.append(str(elem))
            else:                          # e.g. <br/> or <em>…</em>
                definition_parts.append(elem.get_text(" ", strip=True))
        definition = clean_text(" ".join(definition_parts))

        # guard against empty definition (shouldn’t happen)
        if term and definition:
            pairs.append((term, definition))

    return pairs

def main(out_path: Path = Path("cardio_dictionary.csv")) -> None:
    print("Downloading dictionary page …")
    soup = BeautifulSoup(requests.get(BASE_URL, timeout=30).text, "html.parser")

    all_pairs = []
    for tab_id in tqdm(TAB_IDS, desc="Parsing tabs"):
        all_pairs.extend(extract_pairs(soup, tab_id))

    # Deduplicate on term (keep the first definition seen)
    seen = {}
    for term, definition in all_pairs:
        if term not in seen:
            seen[term] = definition

    df = pd.DataFrame(
        sorted(seen.items(), key=lambda x: x[0].lower()),
        columns=["term", "definition"],
    )
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} terms to {out_path.absolute()}")

if __name__ == "__main__":
    main()
