#!/usr/bin/env python3
"""
Auto-update papers.bib from Google Scholar via Semantic Scholar API.

How it works:
  1. Fetches all papers for the author from Semantic Scholar (free, no key needed).
  2. Compares against existing papers.bib to find new entries.
  3. For each new paper, builds a clean BibTeX entry (with arXiv PDF link if available).
  4. Appends new entries to _bibliography/papers.bib.

Usage:
    python3 _scripts/update_publications.py
"""

import re
import sys
import time
from pathlib import Path

import requests

# ── Config ─────────────────────────────────────────────────────────────────────
SCHOLAR_ID     = "PKjnTR4AAAAJ"   # from scholar.google.com/citations?user=THIS
SS_AUTHOR_URL  = "https://api.semanticscholar.org/graph/v1/author/search"
SS_PAPERS_URL  = "https://api.semanticscholar.org/graph/v1/author/{}/papers"
SS_PAPER_URL   = "https://api.semanticscholar.org/graph/v1/paper/{}"

PAPER_FIELDS   = "title,authors,year,venue,externalIds,openAccessPdf,publicationTypes,publicationDate"
AUTHOR_NAME    = "Sadjad Alikhani"

BIB_FILE       = Path(__file__).parent.parent / "_bibliography" / "papers.bib"

# ── Helpers ────────────────────────────────────────────────────────────────────
def ss_get(url: str, params: dict = {}) -> dict:
    """GET with retry on rate-limit."""
    for attempt in range(4):
        r = requests.get(url, params=params, timeout=20,
                         headers={"User-Agent": "publication-updater/1.0"})
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"  Rate limited — waiting {wait}s …")
            time.sleep(wait)
        else:
            print(f"  HTTP {r.status_code} for {url}")
            return {}
    return {}


def find_ss_author_id(name: str) -> str | None:
    """Search Semantic Scholar for author by name, return their SS author ID."""
    data = ss_get(SS_AUTHOR_URL, {"query": name, "limit": 5,
                                   "fields": "name,affiliations,paperCount"})
    for author in data.get("data", []):
        if name.lower() in author.get("name", "").lower():
            return author["authorId"]
    return None


def fetch_all_papers(author_id: str) -> list[dict]:
    """Fetch all papers for a Semantic Scholar author ID."""
    papers, offset, limit = [], 0, 100
    while True:
        data = ss_get(SS_PAPERS_URL.format(author_id),
                      {"fields": PAPER_FIELDS, "limit": limit, "offset": offset})
        batch = data.get("data", [])
        if not batch:
            break
        papers.extend(batch)
        if data.get("next") is None or len(batch) < limit:
            break
        offset += limit
        time.sleep(0.5)
    return papers


def existing_titles(bib_text: str) -> set[str]:
    """Extract normalised titles already in the bib file."""
    titles = re.findall(r'\btitle\s*=\s*[{"](.+?)[}"]', bib_text, re.IGNORECASE)
    return {t.strip().lower() for t in titles}


def existing_arxiv_ids(bib_text: str) -> set[str]:
    return set(re.findall(r'arXiv:(\d{4}\.\d{4,5})', bib_text, re.IGNORECASE))


def make_cite_key(paper: dict) -> str:
    first_author = (paper.get("authors") or [{}])[0].get("name", "unknown")
    last = first_author.split()[-1].lower()
    year = paper.get("year") or "0000"
    slug  = re.sub(r'\W+', '', paper.get("title", "untitled").split()[0].lower())
    return f"{last}{year}{slug}"


def make_bibtex(paper: dict) -> str:
    title   = paper.get("title", "Unknown Title")
    authors = " and ".join(a["name"] for a in (paper.get("authors") or []))
    year    = paper.get("year") or ""
    venue   = paper.get("venue") or ""
    ext     = paper.get("externalIds") or {}
    arxiv   = ext.get("ArXiv", "")
    doi     = ext.get("DOI", "")
    pdf_obj = paper.get("openAccessPdf") or {}
    pdf_url = pdf_obj.get("url") or (f"https://arxiv.org/pdf/{arxiv}" if arxiv else "")
    url     = f"https://arxiv.org/abs/{arxiv}" if arxiv else (f"https://doi.org/{doi}" if doi else "")
    pub_types = paper.get("publicationTypes") or []

    # Choose entry type
    if any(t in pub_types for t in ["JournalArticle"]):
        entry_type = "article"
        venue_field = f"  journal={{{venue}}},\n" if venue else ""
    elif any(t in pub_types for t in ["Conference", "ConferencePaper"]):
        entry_type = "inproceedings"
        venue_field = f"  booktitle={{{venue}}},\n" if venue else ""
    else:
        entry_type = "article"   # default to article for arXiv preprints
        venue_field = f"  journal={{{venue or 'arXiv preprint'}}},\n"

    key = make_cite_key(paper)
    abbr = "arXiv" if arxiv and not doi else (venue[:8] if venue else "")

    lines = [f"@{entry_type}{{{key},"]
    if abbr:
        lines.append(f"  abbr={{{abbr}}},")
    lines.append(f"  title={{{title}}},")
    lines.append(f"  author={{{authors}}},")
    if venue_field.strip():
        lines.append(f"  {venue_field.strip()}")
    if year:
        lines.append(f"  year={{{year}}},")
    if url:
        lines.append(f"  url={{{url}}},")
    if pdf_url:
        lines.append(f"  pdf={{{pdf_url}}},")
    if doi:
        lines.append(f"  doi={{{doi}}},")
    lines.append("  selected={false}")
    lines.append("}")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Looking up '{AUTHOR_NAME}' on Semantic Scholar …")
    author_id = find_ss_author_id(AUTHOR_NAME)
    if not author_id:
        sys.exit(f"Could not find author '{AUTHOR_NAME}' on Semantic Scholar.")
    print(f"  Found author ID: {author_id}")

    print("Fetching paper list …")
    papers = fetch_all_papers(author_id)
    print(f"  Found {len(papers)} paper(s) on Semantic Scholar.")

    bib_text = BIB_FILE.read_text(encoding="utf-8")
    known_titles  = existing_titles(bib_text)
    known_arxivs  = existing_arxiv_ids(bib_text)

    new_entries = []
    for paper in papers:
        title_norm = (paper.get("title") or "").strip().lower()
        arxiv_id   = (paper.get("externalIds") or {}).get("ArXiv", "")

        # Skip if already in bib
        if title_norm in known_titles:
            continue
        if arxiv_id and arxiv_id in known_arxivs:
            continue
        # Skip papers with no real metadata
        if not paper.get("title") or not paper.get("authors"):
            continue

        print(f"  + New: {paper['title'][:70]}")
        new_entries.append(make_bibtex(paper))
        time.sleep(0.3)

    if not new_entries:
        print("No new publications found — papers.bib is up to date.")
        return

    separator = "\n\n% ── Auto-added ──────────────────────────────────────\n\n"
    BIB_FILE.write_text(
        bib_text.rstrip() + separator + "\n\n".join(new_entries) + "\n",
        encoding="utf-8"
    )
    print(f"\n✓ Added {len(new_entries)} new entry/entries to {BIB_FILE}")


if __name__ == "__main__":
    main()
