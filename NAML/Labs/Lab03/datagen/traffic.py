#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import datetime as dt
import time
from urllib.parse import quote

# ----------------------------
# 1. Setup
# ----------------------------
WIKI_REST_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
PROJECT = "en.wikipedia.org"
ACCESS = "all-access"   # desktop, mobile, etc.
AGENT = "all-agents"    # all users

# Use the last 30 days
end_date = dt.date.today() - dt.timedelta(days=1)
start_date = end_date - dt.timedelta(days=30)
start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

# ----------------------------
# 2. Read node list
# ----------------------------
nodes = pd.read_csv("nodes.csv")
titles = nodes["node"].tolist()

traffic_data = []

# ----------------------------
# 3. Fetch traffic per article
# ----------------------------
session = requests.Session()
session.headers.update({"User-Agent": "MatteoCaldana/1.0 Wikipedia Graph Builder - Teaching at Polimi"})

for i, title in enumerate(titles, 1):
    title_encoded = quote(title.replace(" ", "_"))
    url = f"{WIKI_REST_API}/{PROJECT}/{ACCESS}/{AGENT}/{title_encoded}/daily/{start_str}/{end_str}"
    
    print(f"[{i}/{len(titles)}] Fetching: {title}")
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            print(f" Skipped (status {r.status_code})")
            continue
        data = r.json()
        views = [d["views"] for d in data.get("items", [])]
        total_views = sum(views)
        traffic_data.append({"node": title, "traffic": total_views})
    except Exception as e:
        print(f" Error for {title}: {e}")
        continue

    time.sleep(0.2)  # be polite to the API

# ----------------------------
# 4. Save to CSV
# ----------------------------
traffic_df = pd.DataFrame(traffic_data)
traffic_df.to_csv("traffic.csv", index=False)

print(f"\nSaved {len(traffic_df)} entries to traffic.csv")
