import re
import time
import json
import yaml
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# -----------------------------
# Helpers
# -----------------------------

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

def extract_doi(item: Dict[str, Any]) -> Optional[str]:
    """
    Tries to extract a DOI from:
      - item["id"] like "doi:10.1111/jofi.13260"
      - item["url"] like "https://doi.org/10.1111/jofi.13260"
    Returns normalized DOI (lowercase), or None.
    """
    for key in ("id", "url"):
        val = item.get(key)
        if not val or not isinstance(val, str):
            continue
        m = DOI_RE.search(val)
        if m:
            return m.group(0).lower()
    return None

def inverted_index_to_text(inv: Dict[str, Any]) -> str:
    """
    OpenAlex abstracts are often provided as abstract_inverted_index:
      { "word": [pos1, pos2, ...], ... }
    Reconstructs the abstract string.
    """
    if not isinstance(inv, dict) or not inv:
        return ""

    # Determine length
    max_pos = -1
    for positions in inv.values():
        if isinstance(positions, list) and positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""

    tokens = [""] * (max_pos + 1)
    for word, positions in inv.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int) and 0 <= p < len(tokens):
                tokens[p] = word

    # Join with spaces, with a small cleanup
    text = " ".join(t for t in tokens if t)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # remove space before punctuation
    return text.strip()

def fetch_openalex_work_by_doi(doi: str, session: requests.Session, email: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch a work from OpenAlex by DOI.
    Endpoint: https://api.openalex.org/works/https://doi.org/{doi}
    """
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    params = {}
    if email:
        # OpenAlex recommends including an email in requests
        params["mailto"] = email

    r = session.get(url, params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

# -----------------------------
# Main
# -----------------------------

def main(
    yml_path: str,
    out_dir: str = "openalex_abstracts_out",
    sleep_seconds: float = 0.2,
    email_for_openalex: Optional[str] = None,
    write_updated_yaml: bool = True
):
    yml_path = Path(yml_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with yml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "main" not in raw or not isinstance(raw["main"], list):
        raise ValueError(
            "Expected YAML of the form: { main: [ ... ] }. "
            f"Got top-level type={type(raw)} keys={list(raw.keys()) if isinstance(raw, dict) else None}"
        )

    data = raw["main"]


    session = requests.Session()
    session.headers.update({"User-Agent": "doi-abstract-fetcher/1.0"})

    results = []
    updated_data = []

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            updated_data.append(item)
            continue

        doi = extract_doi(item)
        title = (item.get("title") or "").strip()
        venue = (item.get("venue") or "").strip()
        date = (item.get("date") or "").strip()

        record = {
            "idx": i,
            "doi": doi,
            "title": title,
            "venue": venue,
            "date": date,
            "openalex_id": None,
            "has_abstract": False,
            "abstract": "",
            "error": ""
        }

        if not doi:
            record["error"] = "no_doi_found"
            results.append(record)
            updated_data.append(item)
            continue

        try:
            work = fetch_openalex_work_by_doi(doi, session=session, email=email_for_openalex)
            if work is None:
                record["error"] = "openalex_not_found"
            else:
                record["openalex_id"] = work.get("id")
                inv = work.get("abstract_inverted_index")
                abstract = inverted_index_to_text(inv) if inv else ""
                if abstract:
                    record["has_abstract"] = True
                    record["abstract"] = abstract

                    # Optionally attach abstract to the YAML entry
                    if write_updated_yaml:
                        item = dict(item)  # shallow copy
                        item["abstract"] = abstract
                        item["abstract_source"] = "openalex"
                else:
                    record["error"] = "no_abstract_in_openalex"
        except Exception as e:
            record["error"] = f"exception: {type(e).__name__}: {e}"

        results.append(record)
        updated_data.append(item)

        # be polite to the API
        time.sleep(sleep_seconds)

    # Save JSONL (easy to diff / stream)
    jsonl_path = out_dir / "abstracts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save CSV (easy to browse)
    df = pd.DataFrame(results)
    csv_path = out_dir / "abstracts.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Save updated YAML (optional)
    if write_updated_yaml:
        updated_yml_path = out_dir / f"{yml_path.stem}.with_abstracts.yml"
        with updated_yml_path.open("w", encoding="utf-8") as f:
            out_raw = dict(raw)
            out_raw["main"] = updated_data
            yaml.safe_dump(out_raw, f, sort_keys=False, allow_unicode=True)

    # Simple summary
    n = len(results)
    n_found = sum(1 for r in results if r["has_abstract"])
    print(f"Processed {n} entries. Abstracts found: {n_found}.")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {csv_path}")
    if write_updated_yaml:
        print(f"Wrote: {updated_yml_path}")

if __name__ == "__main__":
    # main("papers.yml", email_for_openalex="you@uni.edu")
    main(r"C:\Users\jonat\Lasso_paper\narratives_construction\_data\reading.yml")
