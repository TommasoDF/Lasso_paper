#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import yaml

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
READING_YML = REPO_ROOT / "_data" / "reading.yml"
CONFIG_YML = REPO_ROOT / "scripts" / "reading_sources.yml"

YAML_KEY = "main"

CROSSREF_BASE = "https://api.crossref.org"
OPENALEX_BASE = "https://api.openalex.org"

USER_AGENT = "reading-list-bot/2.0 (personal academic website; contact: you@example.com)"


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class Candidate:
    id: str
    title: str
    authors: List[str]
    date_ym: Optional[str]
    venue: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    abstract: Optional[str]
    source: str
    raw: Dict[str, Any]


# -----------------------------------------------------------------------------
# YAML IO
# -----------------------------------------------------------------------------
def load_yaml_list(path: Path, key: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not path.exists():
        print(f"[INFO] {path} does not exist; creating new file.")
        return {key: []}, []

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML dict at top level")

    items = data.get(key, [])
    if not isinstance(items, list):
        raise ValueError(f"{path}:{key} must be a YAML list")

    return data, items


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    yaml.safe_dump(
        data,
        path.open("w", encoding="utf-8"),
        sort_keys=False,
        allow_unicode=True,
        width=110,
        default_flow_style=False,
    )


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def norm_doi(doi: str) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I)


def doi_url(doi: str) -> str:
    return f"https://doi.org/{norm_doi(doi)}"


def authors_to_string(authors: List[str]) -> str:
    return " , ".join(a.strip() for a in authors if a and a.strip())


def join_title(t: Any) -> str:
    return " ".join(t) if isinstance(t, list) else str(t or "").strip()


def dateparts_to_ym(dp: Any) -> Optional[str]:
    try:
        y = int(dp[0][0])
        m = int(dp[0][1]) if len(dp[0]) > 1 else 1
        return f"{y:04d}-{m:02d}"
    except Exception:
        return None


def strip_tags(s: str) -> str:
    # crude, but good enough for Crossref JATS-ish abstracts
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def inverted_index_to_text(inv: Dict[str, Any]) -> str:
    """
    OpenAlex stores abstract as an inverted index: {word: [pos, pos, ...], ...}
    Reconstruct the original token sequence.
    """
    if not inv or not isinstance(inv, dict):
        return ""
    try:
        max_pos = max(p for positions in inv.values() for p in positions)
    except ValueError:
        return ""
    tokens = [""] * (max_pos + 1)
    for word, positions in inv.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int) and 0 <= p <= max_pos:
                tokens[p] = word
    return " ".join(t for t in tokens if t)


def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.I) for p in (patterns or [])]


def candidate_text(c: Candidate) -> str:
    # Filter over what we have: title + venue + abstract
    parts = [c.title or "", c.venue or "", c.abstract or ""]
    return "\n".join(p for p in parts if p)


def passes_filter(text: str, include_res: List[re.Pattern], exclude_res: List[re.Pattern]) -> bool:
    if exclude_res and any(rx.search(text) for rx in exclude_res):
        return False
    if include_res and not any(rx.search(text) for rx in include_res):
        return False
    return True


# -----------------------------------------------------------------------------
# HTTP session
# -----------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


# -----------------------------------------------------------------------------
# Crossref
# -----------------------------------------------------------------------------
def crossref_iter_by_issn(
    session: requests.Session,
    issn: str,
    from_date: dt.date,
    until_date: dt.date,
    rows: int = 100,
    cursor: str = "*",
) -> Iterable[Dict[str, Any]]:
    """
    Crossref ingestion:
    - /works endpoint with filter=issn:...
    - cursor pagination
    """
    url = f"{CROSSREF_BASE}/works"

    while True:
        params = {
            "filter": f"issn:{issn},from-pub-date:{from_date.isoformat()},until-pub-date:{until_date.isoformat()}",
            "sort": "published",
            "order": "desc",
            "rows": rows,
            "cursor": cursor,
            "cursor-max": rows,
        }

        r = session.get(url, params=params, timeout=30)
        if r.status_code == 404:
            break
        r.raise_for_status()

        payload = r.json()
        msg = payload.get("message") or {}
        items = msg.get("items") or []
        if not items:
            break

        for it in items:
            if isinstance(it, dict):
                yield it

        nxt = msg.get("next-cursor")
        if not nxt or nxt == cursor:
            break
        cursor = nxt


def crossref_candidates(
    session: requests.Session, issns: List[str], window_days: int, cap: int
) -> List[Candidate]:
    until = dt.date.today()
    since = until - dt.timedelta(days=window_days)
    out: List[Candidate] = []

    for issn in issns:
        print(f"  [Crossref] ISSN {issn}")
        fetched = 0

        try:
            for it in crossref_iter_by_issn(session, issn, since, until, rows=100, cursor="*"):
                doi = it.get("DOI")
                title = join_title(it.get("title"))
                if not doi or not title:
                    continue

                doi_n = norm_doi(str(doi))

                # venue
                venue = ""
                ct = it.get("container-title")
                if isinstance(ct, list) and ct:
                    venue = str(ct[0]).strip()
                elif isinstance(ct, str):
                    venue = ct.strip()

                # authors
                authors: List[str] = []
                for a in it.get("author") or []:
                    if not isinstance(a, dict):
                        continue
                    given = str(a.get("given") or "").strip()
                    family = str(a.get("family") or "").strip()
                    nm = (given + " " + family).strip()
                    if nm:
                        authors.append(nm)

                # date (YYYY-MM)
                date_ym = None
                for k in ("published-online", "published-print", "issued", "created", "deposited"):
                    part = it.get(k)
                    if isinstance(part, dict):
                        ym = dateparts_to_ym(part.get("date-parts"))
                        if ym:
                            date_ym = ym
                            break

                # abstract (sometimes present, often JATS)
                abs_raw = it.get("abstract")
                abstract = None
                if isinstance(abs_raw, str) and abs_raw.strip():
                    abstract = strip_tags(abs_raw)

                out.append(
                    Candidate(
                        id=f"doi:{doi_n}",
                        title=title,
                        authors=authors,
                        date_ym=date_ym,
                        venue=venue or None,
                        doi=doi_n,
                        url=doi_url(doi_n),
                        abstract=abstract,
                        source="crossref",
                        raw=it,
                    )
                )

                fetched += 1
                if fetched >= cap:
                    break

            print(f"    → fetched {fetched} candidates")
        except requests.HTTPError as e:
            print(f"    → HTTP error for ISSN {issn}: {e} (skipping)")
            continue

    return out


# -----------------------------------------------------------------------------
# OpenAlex
# -----------------------------------------------------------------------------
def openalex_candidates(
    session: requests.Session, issns: List[str], window_days: int, cap: int
) -> List[Candidate]:
    until = dt.date.today()
    since = until - dt.timedelta(days=window_days)
    out: List[Candidate] = []

    for issn in issns:
        print(f"  [OpenAlex] ISSN {issn}")

        r = session.get(f"{OPENALEX_BASE}/sources/issn:{issn}", timeout=30)
        if r.status_code == 404:
            print("    → no source found")
            continue
        r.raise_for_status()

        source_id = (r.json() or {}).get("id")
        if not source_id:
            print("    → source missing id")
            continue

        fetched = 0
        page = 1
        total_pages_scanned = 0

        api_filter = (
            f"primary_location.source.id:{source_id},"
            f"from_publication_date:{since.isoformat()},"
            f"to_publication_date:{until.isoformat()}"
        )

        while fetched < cap:
            r = session.get(
                f"{OPENALEX_BASE}/works",
                params={
                    "filter": api_filter,
                    "sort": "publication_date:desc",
                    "per-page": 200,
                    "page": page,
                },
                timeout=30,
            )
            r.raise_for_status()

            results = (r.json() or {}).get("results", []) or []
            if not results:
                break

            total_pages_scanned += 1
            added_this_page = 0

            for it in results:
                pub = it.get("publication_date")
                if not pub:
                    continue
                if pub < since.isoformat() or pub > until.isoformat():
                    continue

                doi = it.get("doi")
                title = (it.get("title") or "").strip()
                if not title:
                    continue

                # abstract from inverted index
                abstract = inverted_index_to_text(it.get("abstract_inverted_index") or {})
                abstract = abstract.strip() or None

                out.append(
                    Candidate(
                        id=f"doi:{norm_doi(doi)}" if doi else f"openalex:{it.get('id')}",
                        title=title,
                        authors=[
                            a["author"]["display_name"]
                            for a in (it.get("authorships") or [])
                            if isinstance(a, dict) and a.get("author")
                        ],
                        date_ym=pub[:7],
                        venue=((it.get("primary_location") or {}).get("source") or {}).get("display_name"),
                        doi=norm_doi(doi) if doi else None,
                        url=doi_url(doi) if doi else it.get("id"),
                        abstract=abstract,
                        source="openalex",
                        raw=it,
                    )
                )

                fetched += 1
                added_this_page += 1
                if fetched >= cap:
                    break

            if added_this_page == 0:
                break

            page += 1
            if total_pages_scanned >= 50:
                print("    → stopping after 50 pages (safety cap)")
                break

        print(f"    → fetched {fetched} candidates")

    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    print("[START] Updating reading list")

    cfg = load_config(CONFIG_YML)

    # filtering
    flt = cfg.get("filter", {}) or {}
    include_res = compile_patterns(flt.get("include_patterns", []) or [])
    exclude_res = compile_patterns(flt.get("exclude_patterns", []) or [])

    window_days = int(cfg.get("window_days", 14))
    cap = int(cfg.get("max_new_per_journal", 30))

    data, items = load_yaml_list(READING_YML, YAML_KEY)
    seen = {
        it.get("id") or f"url:{it.get('url')}"
        for it in items
        if isinstance(it, dict)
    }

    session = make_session()

    added_total = 0
    filtered_total = 0

    for j in (cfg.get("journals") or []):
        name = j.get("name", "unknown journal")
        issns = j.get("issn", []) or []
        providers = j.get("providers", ["crossref"]) or ["crossref"]

        if isinstance(issns, str):
            issns = [issns]

        print(f"\n[Journal] {name}")

        candidates: List[Candidate] = []
        if "crossref" in providers:
            candidates += crossref_candidates(session, issns, window_days, cap)
        if "openalex" in providers:
            candidates += openalex_candidates(session, issns, window_days, cap)

        print(f"  total candidates: {len(candidates)}")

        new_here = 0
        filtered_here = 0

        for c in candidates:
            if c.id in seen:
                continue

            text = candidate_text(c)
            if not passes_filter(text, include_res, exclude_res):
                filtered_here += 1
                continue

            items.insert(
                0,
                {
                    "title": c.title,
                    "authors": authors_to_string(c.authors),
                    "date": c.date_ym or f"{dt.date.today().year}-01",
                    "venue": c.venue or "",
                    "url": c.url or "",
                    "read": False,
                    "id": c.id,
                    "source": c.source,
                },
            )

            seen.add(c.id)
            new_here += 1
            added_total += 1

        filtered_total += filtered_here

        if new_here:
            print(f"  → added {new_here} new papers")
        else:
            print("  → no new papers found")
        if filtered_here:
            print(f"  → filtered out {filtered_here} candidates")

    if added_total:
        data[YAML_KEY] = items
        dump_yaml(READING_YML, data)
        print(f"\n[DONE] Added {added_total} total papers (filtered out {filtered_total})")
    else:
        print(f"\n[DONE] No changes (filtered out {filtered_total})")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
