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

import os
import re
import subprocess
import sys
import tempfile
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
RESUME_FILE    = Path(__file__).parent.parent / "assets" / "json" / "resume.json"

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


# ── Resume JSON sync ───────────────────────────────────────────────────────────
def sync_resume(papers: list[dict], known_titles: set[str]) -> int:
    """Add new papers to the publications array in resume.json. Returns count added."""
    import json

    resume = json.loads(RESUME_FILE.read_text(encoding="utf-8"))
    existing = {p["name"].strip().lower() for p in resume.get("publications", [])}

    added = 0
    for paper in sorted(papers, key=lambda p: p.get("year") or 0, reverse=True):
        title = (paper.get("title") or "").strip()
        if not title or title.lower() in existing:
            continue

        authors = paper.get("authors") or []
        author_str = ", ".join(
            # Format as "F. Lastname"
            (f"{a['name'].split()[0][0]}. {a['name'].split()[-1]}" if len(a["name"].split()) > 1 else a["name"])
            for a in authors
        )

        ext    = paper.get("externalIds") or {}
        arxiv  = ext.get("ArXiv", "")
        doi    = ext.get("DOI", "")
        url    = f"https://arxiv.org/abs/{arxiv}" if arxiv else (f"https://doi.org/{doi}" if doi else "#")
        venue  = paper.get("venue") or "arXiv preprint"
        year   = paper.get("year") or ""
        date   = f"{year}-01-01" if year else ""

        entry = {
            "name": title,
            "publisher": venue,
            "releaseDate": date,
            "url": url,
            "summary": f"{author_str}.",
        }

        resume.setdefault("publications", []).insert(0, entry)
        existing.add(title.lower())
        print(f"  + resume.json: {title[:70]}")
        added += 1

    if added:
        RESUME_FILE.write_text(
            json.dumps(resume, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
    return added


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

    if new_entries:
        separator = "\n\n% ── Auto-added ──────────────────────────────────────\n\n"
        BIB_FILE.write_text(
            bib_text.rstrip() + separator + "\n\n".join(new_entries) + "\n",
            encoding="utf-8"
        )
        print(f"\n✓ Added {len(new_entries)} new entry/entries to papers.bib")
    else:
        print("No new publications found — papers.bib is up to date.")

    # Sync all fetched papers into resume.json as well
    print("\nSyncing resume.json …")
    n = sync_resume(papers, known_titles)
    if n == 0:
        print("  resume.json is already up to date.")
    else:
        print(f"  ✓ Added {n} entry/entries to resume.json")

    # Update LaTeX CV if new papers were found
    if new_entries or n > 0:
        print("\nUpdating cv-latex repo …")
        update_cv_latex(papers)


# ── CV LaTeX updater ───────────────────────────────────────────────────────────
def update_cv_latex(papers: list[dict]) -> None:
    """Clone cv-latex, ask GPT-4o-mini to update publications section, push back."""
    import json

    gh_pat      = os.environ.get("GH_PAT", "")
    openai_key  = os.environ.get("OPENAI_API_KEY", "")
    cv_repo     = "sadjadalikhani/cv-latex"
    tex_file    = "CV_Sadjad_Alikhani.tex"

    if not gh_pat:
        print("  Skipping cv-latex update — GH_PAT not set.")
        return
    if not openai_key:
        print("  Skipping cv-latex update — OPENAI_API_KEY not set.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clone_url = f"https://x-access-token:{gh_pat}@github.com/{cv_repo}.git"

        print("  Cloning cv-latex …")
        subprocess.run(["git", "clone", "--depth=1", clone_url, str(tmp / "repo")],
                       check=True, capture_output=True)

        repo_dir = tmp / "repo"
        tex_path = repo_dir / tex_file
        if not tex_path.exists():
            print(f"  ERROR: {tex_file} not found in cv-latex repo.")
            return

        current_tex = tex_path.read_text(encoding="utf-8")

        # Build list of all papers for the prompt
        paper_list = []
        for p in sorted(papers, key=lambda x: x.get("year") or 0, reverse=True):
            title   = p.get("title", "")
            authors = ", ".join(a["name"] for a in (p.get("authors") or []))
            venue   = p.get("venue") or "arXiv preprint"
            year    = p.get("year") or ""
            ext     = p.get("externalIds") or {}
            arxiv   = ext.get("ArXiv", "")
            arxiv_str = f" arXiv:{arxiv}" if arxiv else ""
            paper_list.append(f'- "{title}" by {authors}. {venue}{arxiv_str}, {year}.')
        papers_text = "\n".join(paper_list)

        prompt = f"""You are a LaTeX CV editor. Below is my current CV LaTeX source.
Your task: update ONLY the Publications section to include all papers listed below.
Keep the exact same LaTeX formatting style as existing entries (\\item \\justifying ...).
Keep existing entries unchanged. Add missing ones in reverse chronological order (newest first).
Return the COMPLETE updated LaTeX file, nothing else — no explanation, no markdown fences.

COMPLETE PAPER LIST (use these as the ground truth):
{papers_text}

CURRENT LaTeX CV:
{current_tex}
"""

        print("  Calling GPT-4o-mini …")
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_key}",
                     "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini",
                  "max_tokens": 16000,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"  OpenAI error {resp.status_code}: {resp.text[:300]}")
            return

        updated_tex = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip any accidental markdown fences
        if updated_tex.startswith("```"):
            updated_tex = re.sub(r'^```[^\n]*\n', '', updated_tex)
            updated_tex = re.sub(r'\n```$', '', updated_tex.rstrip())

        tex_path.write_text(updated_tex, encoding="utf-8")

        # Commit and push
        git_env = {"GIT_AUTHOR_NAME": "github-actions[bot]",
                   "GIT_AUTHOR_EMAIL": "github-actions[bot]@users.noreply.github.com",
                   "GIT_COMMITTER_NAME": "github-actions[bot]",
                   "GIT_COMMITTER_EMAIL": "github-actions[bot]@users.noreply.github.com",
                   **dict(__import__("os").environ)}

        subprocess.run(["git", "config", "user.email",
                        "github-actions[bot]@users.noreply.github.com"],
                       cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"],
                       cwd=repo_dir, check=True)
        subprocess.run(["git", "add", tex_file], cwd=repo_dir, check=True)

        diff = subprocess.run(["git", "diff", "--staged", "--quiet"],
                              cwd=repo_dir).returncode
        if diff == 0:
            print("  cv-latex tex unchanged — no commit needed.")
            return

        subprocess.run(["git", "commit", "-m",
                        f"cv: auto-update publications {time.strftime('%Y-%m-%d')}"],
                       cwd=repo_dir, check=True, env=git_env)
        subprocess.run(["git", "push"], cwd=repo_dir, check=True)
        print("  ✓ cv-latex updated and pushed — PDF will recompile automatically.")


if __name__ == "__main__":
    main()
