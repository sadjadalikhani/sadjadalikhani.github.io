#!/usr/bin/env python3
"""
Google Scholar citation checker — runs on GitHub Actions every 15 minutes.
Emails alikhani@asu.edu when the citation count increases.
Only commits state to the repo when the count actually changes (avoids noisy
commits on every run).
"""

import os
import json
import smtplib
import subprocess
import sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
SCHOLAR_URL = "https://scholar.google.com/citations?user=PKjnTR4AAAAJ&hl=en"
RECIPIENT   = "alikhani@asu.edu"
SMTP_HOST   = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER   = os.environ.get("SMTP_USER", "")
SMTP_PASS   = os.environ.get("SMTP_PASS", "")

SCRIPT_DIR  = Path(__file__).parent
REPO_DIR    = SCRIPT_DIR.parent          # workspace root for git commands
STATE_FILE  = SCRIPT_DIR / "last_citations.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
# ─────────────────────────────────────────────────────────────────────────────


def fetch_citation_count() -> dict:
    """Return {'total': int, 'h_index': int, 'i10_index': int, 'name': str}."""
    resp = requests.get(SCHOLAR_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    name_el = soup.select_one("#gsc_prf_in")
    name = name_el.get_text(strip=True) if name_el else "Unknown"

    stat_cells = soup.select("td.gsc_rsb_std")
    if len(stat_cells) < 5:
        raise ValueError("Could not parse citation stats — page structure may have changed.")

    total   = int(stat_cells[0].get_text(strip=True))
    h_index = int(stat_cells[2].get_text(strip=True))
    i10     = int(stat_cells[4].get_text(strip=True))

    return {"total": total, "h_index": h_index, "i10_index": i10, "name": name}


def load_state() -> dict | None:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def save_state(data: dict) -> None:
    payload = {**data, "last_updated": datetime.utcnow().isoformat()}
    with open(STATE_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def commit_state(message: str) -> None:
    """Commit the updated state file and push. Git must already be configured."""
    rel = str(STATE_FILE.relative_to(REPO_DIR))
    cmds = [
        ["git", "-C", str(REPO_DIR), "add", rel],
        ["git", "-C", str(REPO_DIR), "commit", "-m", message],
        ["git", "-C", str(REPO_DIR), "push"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"git command failed: {' '.join(cmd)}\n{r.stderr}", file=sys.stderr)
            sys.exit(1)
    print(f"  ✓ State committed: {message}")


def send_email(prev: dict, curr: dict) -> None:
    if not SMTP_USER or not SMTP_PASS:
        print("ERROR: SMTP_USER / SMTP_PASS not set — cannot send email.", file=sys.stderr)
        sys.exit(1)

    delta = curr["total"] - prev["total"]
    subject = f"New citation(s) on Google Scholar (+{delta})"
    body = (
        f"Hi {curr['name']},\n\n"
        f"Your Google Scholar citation count just increased!\n\n"
        f"  Previous total citations : {prev['total']}\n"
        f"  Current  total citations : {curr['total']}  (+{delta})\n"
        f"  h-index                  : {curr['h_index']}\n"
        f"  i10-index                : {curr['i10_index']}\n\n"
        f"View your profile: {SCHOLAR_URL}\n\n"
        f"Checked at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    )

    msg = MIMEMultipart()
    msg["From"]    = SMTP_USER
    msg["To"]      = RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, RECIPIENT, msg.as_string())

    print(f"  ✓ Email sent to {RECIPIENT}: {subject}")


def main() -> None:
    print(f"[{datetime.utcnow().isoformat()}] Checking Google Scholar citations…")

    curr = fetch_citation_count()
    print(f"  Current citations: {curr['total']} (h={curr['h_index']}, i10={curr['i10_index']})")

    prev = load_state()

    if prev is None:
        print("  No previous state found — saving baseline (no email sent).")
        save_state(curr)
        commit_state(f"gs-notif: baseline {curr['total']} citations")
        return

    if curr["total"] > prev["total"]:
        print(f"  New citations detected: {prev['total']} → {curr['total']}")
        send_email(prev, curr)
        save_state(curr)
        commit_state(f"gs-notif: citations {prev['total']} → {curr['total']}")
    else:
        # No change — do NOT commit; avoids a noisy commit every 15 minutes.
        print(f"  No change (still {curr['total']}).")


if __name__ == "__main__":
    main()
