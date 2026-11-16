import os
import random
import re
import time
import pandas as pd
import requests

# Insert your free key from https://openalex.org/settings/api
API_KEY = os.getenv("OPENALEX_API_KEY", "Input_your_API_key")

RANDOM_SEED = 42
TARGET_PER_GROUP = 120  # Target size for each group (240 articles in total)
MAX_POOL_SIZE = 350     # Stop collecting the pool once 350 valid articles are reached

PHILOLOGY_JOURNAL_KEYWORDS = [
    "language", "linguistic", "pragmatic", "discourse", "philolog",
    "communication", "speech", "semiotic", "text", "rhetoric",
    "english", "stylistic", "translation", "literary", "literature",
    "applied", "sociolinguistic", "multilingual", "intercultural",
    "dialogue", "conversation", "glossa", "lingua", "system",
    "syntax", "semantics", "grammar", "slovo", "vestnik"
]

EXCLUDE_JOURNAL_KEYWORDS = [
    "plos", "scientific reports", "frontiers in psychology", "animals",
    "medicine", "medical", "clinical", "health", "nursing", "surgery",
    "psychiatry", "neuro", "brain", "computer", "engineering", "ieee",
    "robot", "artificial intelligence", "ai & society", "big data",
    "sustainability", "physics", "biology", "chemistry", "environment"
]

def is_philology_journal(journal_name):
    if not journal_name:
        return False
    j_low = journal_name.lower()
    for bad in EXCLUDE_JOURNAL_KEYWORDS:
        if bad in j_low:
            return False
    for good in PHILOLOGY_JOURNAL_KEYWORDS:
        if good in j_low:
            return True
    return False

def clean_english_abstract(inverted_index):
    if not inverted_index:
        return None
    word_index = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_index.append((pos, word))
    word_index.sort()
    text = " ".join([word for pos, word in word_index])

    # Remove the "Abstract" header and leading special characters
    text = re.sub(r'^(abstract|ABSTRACT|Abstract)[:\.\-\s]+', '', text).strip()
    text = re.sub(r'^[^\w\(\["\']+', '', text).strip()

    # Keep the longest Latin-script segment
    parts = re.split(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', text)
    english_part = max(parts, key=len).strip()

    # Strip Highlights bullet points and trailing Keywords
    if '•' in english_part:
        english_part = english_part.split('•')[0].strip()

    cut_markers = [
        r'\bKeywords\s*:',
        r'\bKey words\s*:',
        r'\bWHAT THIS PAPER ADDS\s*:',
        r'\bCopyright\s*©'
    ]
    for marker in cut_markers:
        english_part = re.split(marker, english_part, flags=re.IGNORECASE)[0].strip()

    english_part = re.sub(r'^[\.\,\;\:\)\]\-\s]+', '', english_part).strip()

    # Check for English language
    en_markers = re.findall(r'\b(the|and|of|to|in|is|that|for|this|study|research|analysis|with|as)\b', english_part.lower())
    if len(en_markers) < 10:
        return None

    ascii_chars = sum(1 for c in english_part if ord(c) < 128)
    if len(english_part) == 0 or (ascii_chars / len(english_part)) < 0.92:
        return None

    english_part = re.sub(r'\s+', ' ', english_part).strip()

    # Word count threshold for a scientific abstract (100 - 320 words)
    word_count = len(english_part.split())
    if word_count < 100 or word_count > 320:
        return None

    return english_part

def fetch_philology_group(filter_str, group_name, min_fwci=0.0):
    if API_KEY == "Input_your_API_key" or not API_KEY.strip():
        raise ValueError("Please specify your API_KEY from https://openalex.org at the top of the script!")

    api_url = "https://api.openalex.org/works"
    cursor = "*"
    items_list = []
    seen_dois = set()

    print(f"\nCollecting OpenAlex articles (2021–2023): {group_name}...")
    while cursor and len(items_list) < MAX_POOL_SIZE:
        query_params = {
            "filter": filter_str,
            "per-page": 200,
            "cursor": cursor,
            "api_key": API_KEY
        }
        resp = requests.get(api_url, params=query_params, timeout=25)
        if resp.status_code != 200:
            print(f"OpenAlex API error (code {resp.status_code}): {resp.text[:200]}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for item in results:
            field_name = item.get("primary_topic", {}).get("field", {}).get("display_name", "")
            if field_name not in ["Social Sciences", "Arts and Humanities"]:
                continue

            fwci = item.get("fwci") or 0.0
            if fwci < min_fwci:
                continue

            loc = item.get("primary_location") or {}
            source = loc.get("source") or {}
            journal_name = source.get("display_name", "")
            if not is_philology_journal(journal_name):
                continue

            abstract = clean_english_abstract(item.get("abstract_inverted_index"))
            if not abstract:
                continue

            raw_doi = item.get("doi")
            if raw_doi:
                source_id = raw_doi.replace("https://doi.org", "").lstrip("/")
            else:
                oa_id = item.get("id", "").replace("https://openalex.org", "")
                source_id = f"OpenAlex ID: {oa_id}"

            if source_id in seen_dois:
                continue
            seen_dois.add(source_id)

            items_list.append({
                "Source / DOI": source_id,
                "Journal": journal_name,
                "Citation": item.get("cited_by_count", 0),
                "FWCI": round(fwci, 3),
                "Publishing Year": item.get("publication_year"),
                "Text Abstract": abstract
            })

            if len(items_list) >= MAX_POOL_SIZE:
                break

        cursor = data.get("meta", {}).get("next_cursor")
        print(f"Domain-relevant articles found ({group_name}): {len(items_list)}")
        time.sleep(0.15)

    return items_list

if __name__ == "__main__":
    # 1. Highly cited articles for 2021-2023 (Citations >= 15, FWCI >= 1.5)
    cited_pool = fetch_philology_group(
        "primary_topic.id:T10383,publication_year:2021-2023,type:article,cited_by_count:15-300",
        "High_Status (2021-2023, Citations >= 15)",
        min_fwci=1.5
    )

    # 2. Uncited articles for 2021-2023 (Citations = 0)
    uncited_pool = fetch_philology_group(
        "primary_topic.id:T10383,publication_year:2021-2023,type:article,cited_by_count:0",
        "Low_Status (2021-2023, Citations = 0)",
        min_fwci=0.0
    )

    random.seed(RANDOM_SEED)
    n_final = min(TARGET_PER_GROUP, len(cited_pool), len(uncited_pool))

    if n_final == 0:
        print("Failed to collect articles. Check the filter parameters or API key.")
        exit()

    sampled_cited = random.sample(cited_pool, n_final)
    sampled_uncited = random.sample(uncited_pool, n_final)
    sampled_cited.sort(key=lambda x: x["Citation"], reverse=True)

    final_rows = []
    for idx, item in enumerate(sampled_cited, start=1):
        final_rows.append({
            "ID text": f"Cited_{idx:03d}",
            "Group (Factor)": "High_Status",
            "Journal": item["Journal"],
            "Source / DOI": item["Source / DOI"],
            "Citation": item["Citation"],
            "FWCI": item["FWCI"],
            "Publishing Year": item["Publishing Year"],
            "Text Abstract": item["Text Abstract"]
        })

    for idx, item in enumerate(sampled_uncited, start=1):
        final_rows.append({
            "ID text": f"Uncited_{idx:03d}",
            "Group (Factor)": "Low_Status",
            "Journal": item["Journal"],
            "Source / DOI": item["Source / DOI"],
            "Citation": item["Citation"],
            "FWCI": item["FWCI"],
            "Publishing Year": item["Publishing Year"],
            "Text Abstract": item["Text Abstract"]
        })

    df = pd.DataFrame(final_rows)
    output_filename = "OpenAlex_Philology_2021_2023.xlsx"
    df.to_excel(output_filename, index=False)
    print(f"\nDone! Created file {output_filename}: {n_final} + {n_final} = {len(df)} OpenAlex articles.")
