#!/usr/bin/env python3

import requests
import pandas as pd
import time
import logging
from collections import deque

# -------- Configuration --------
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "MatteoCaldana/1.0 Wikipedia Graph Builder - Teaching at Polimi"
REQUESTS_TIMEOUT = 30  # seconds
MAX_RETRIES = 6
BACKOFF_FACTOR = 1.5
SLEEP_BETWEEN_REQUESTS = 0.4  # polite delay
MAX_PAGES = None  # set to int to limit pages collected, or None for no limit
FOLLOW_SUBCATEGORIES = False  # set True to recursively include subcategories
# -------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def safe_get(params):
    """GET with retry + exponential backoff and JSON validation."""
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            r = session.get(WIKI_API, params=params, timeout=REQUESTS_TIMEOUT)
        except requests.RequestException as e:
            wait = (BACKOFF_FACTOR ** attempt)
            logging.warning("Request exception: %s - retrying after %.1fs", e, wait)
            time.sleep(wait)
            attempt += 1
            continue

        # If non-200, back off and retry
        if r.status_code != 200:
            wait = (BACKOFF_FACTOR ** attempt)
            logging.warning("Non-200 status %s for params=%s - retrying after %.1fs", r.status_code, params, wait)
            time.sleep(wait)
            attempt += 1
            continue

        # Try to parse JSON; if fails, show snippet and retry
        try:
            data = r.json()
            return data
        except ValueError:
            # Not JSON (could be HTML error page). Log and retry.
            snippet = r.text[:500].replace("\n", " ")
            logging.warning("Invalid JSON response (first 500 chars): %s", snippet)
            wait = (BACKOFF_FACTOR ** attempt)
            time.sleep(wait)
            attempt += 1
            continue

    raise RuntimeError(f"Failed to get valid JSON after {MAX_RETRIES} attempts for params: {params}")


def get_category_members(category_title):
    """
    Returns list of page titles in the category (ns=0) and list of subcategories (ns=14)
    """
    members = []
    subcats = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_title if category_title.startswith("Category:") else f"Category:{category_title}",
            "cmlimit": "max",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = safe_get(params)
        q = data.get("query", {})
        for item in q.get("categorymembers", []):
            ns = item.get("ns", -1)
            title = item.get("title")
            if ns == 0:
                members.append(title)
            elif ns == 14:
                subcats.append(title)
        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if MAX_PAGES and len(members) >= MAX_PAGES:
            break

    return members, subcats


def get_outgoing_links(title):
    """Return list of linked page titles (namespace 0) from the given page title, following plcontinue."""
    links = []
    plcontinue = None
    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "plnamespace": 0,
            "pllimit": "max",
            "format": "json",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue

        data = safe_get(params)
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            for l in page.get("links", []):
                links.append(l.get("title"))
        cont = data.get("continue", {})
        plcontinue = cont.get("plcontinue")
        if not plcontinue:
            break
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return links


def normalize_title(title):
    """Normalize title returned by API (it is usually normalized already)."""
    # API returns normalized titles; we will just return the title as-is.
    return title


def build_graph_from_category(root_category, limit_pages=None, follow_subcats=False):
    """
    Builds nodes and directed edges (source->target) where source and target are article titles.
    If follow_subcats is True, recursively traverse subcategories (breadth-first).
    """
    # BFS over categories if follow_subcats True, else single category
    categories_to_visit = deque([root_category])
    all_pages = []
    all_subcats_visited = set()

    while categories_to_visit:
        cat = categories_to_visit.popleft()
        logging.info("Collecting category members for: %s", cat)
        pages, subcats = get_category_members(cat)
        logging.info("Found %d pages and %d subcategories in %s", len(pages), len(subcats), cat)
        for p in pages:
            if p not in all_pages:
                all_pages.append(p)
                if limit_pages and len(all_pages) >= limit_pages:
                    break
        if follow_subcats:
            for sc in subcats:
                if sc not in all_subcats_visited:
                    all_subcats_visited.add(sc)
                    categories_to_visit.append(sc)
        if limit_pages and len(all_pages) >= limit_pages:
            break
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    logging.info("Total pages collected: %d", len(all_pages))
    page_set = set(all_pages)

    edges = []
    # For each page, get outgoing links and keep only those that point to pages in our set
    for i, page in enumerate(all_pages, 1):
        logging.info("[%d/%d] Fetching links from: %s", i, len(all_pages), page)
        try:
            out_links = get_outgoing_links(page)
        except RuntimeError as e:
            logging.error("Failed to fetch links for %s: %s", page, e)
            continue
        # Normalize and filter
        for t in out_links:
            nt = normalize_title(t)
            if nt in page_set:
                edges.append((page, nt))
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    nodes_df = pd.DataFrame({"node": all_pages})
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    return nodes_df, edges_df


def main():
    root_category = "Machine_learning"
    limit = MAX_PAGES  # e.g. 200 to limit size
    follow_subcats = FOLLOW_SUBCATEGORIES

    nodes_df, edges_df = build_graph_from_category(root_category, limit_pages=limit, follow_subcats=follow_subcats)

    nodes_df.to_csv("nodes.csv", index=False)
    edges_df.to_csv("edges.csv", index=False)
    logging.info("Saved nodes.csv (%d rows) and edges.csv (%d rows)", len(nodes_df), len(edges_df))


if __name__ == "__main__":
    main()
