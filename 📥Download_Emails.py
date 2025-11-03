import streamlit as st
import requests
from datetime import datetime, time, timedelta
import pandas as pd
import io, zipfile, re
import time as time_module
from typing import Dict, List, Optional
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
API_BASE_URL = "https://api.kit.com/v4"
DEFAULT_PER_PAGE = 200          # reasonable page size to keep payloads smooth
RATE_LIMIT_WINDOW = 60          # seconds
MAX_REQUESTS_PER_WINDOW = 120   # Kit API key auth published limit
GENTLE_PACING_SEC = 0.66        # ~90 req/min steady pace to avoid 429s
MAX_RETRIES = 5                 # for 429 / transient network errors

st.set_page_config(page_title="Kit Email Archive Tool", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _parse_iso_z(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        # normalize 'Z' to UTC tzinfo
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "broadcast"

def in_range(created_at: Optional[str], start_dt: datetime, end_dt: datetime) -> bool:
    dt = _parse_iso_z(created_at)
    return bool(dt and (start_dt <= dt <= end_dt))

def matches_filters(thing: Dict, published_filter: str, public_filter: str) -> bool:
    if published_filter != "All":
        is_published = thing.get("send_at") is not None
        if (published_filter == "Published") != is_published:
            return False
    if public_filter != "All":
        is_public = bool(thing.get("public", False))
        if (public_filter == "Public") != is_public:
            return False
    return True

def save_to_markdown_zip(df: pd.DataFrame) -> bytes:
    """
    Build an in-memory ZIP containing one .md per broadcast with YAML frontmatter
    and the email body (html/plain text if available).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, r in df.iterrows():
            subj = r.get("subject") or r.get("title") or f"Broadcast {r.get('id')}"
            sent = r.get("send_at") or r.get("created_at") or ""
            body = (
                r.get("content")
                or r.get("html_content")
                or r.get("plain_text")
                or r.get("body")
                or ""
            )
            fname = f"{_slug(str(sent)[:10])}-{_slug(str(subj))[:80]}.md"
            md = f"""---
id: {r.get("id")}
subject: "{str(subj).replace('"', "'")}"
send_at: "{sent}"
public: {bool(r.get("public", False))}
---

{body}
"""
            zf.writestr(fname, md)
    buf.seek(0)
    return buf.read()

# ──────────────────────────────────────────────────────────────────────────────
# Rate limiting + API client
# ──────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()

    def wait_if_needed(self):
        now = datetime.now()
        while self.requests and self.requests[0] < now - timedelta(seconds=self.window_seconds):
            self.requests.popleft()
        if len(self.requests) >= self.max_requests:
            oldest = self.requests[0]
            wait = (oldest + timedelta(seconds=self.window_seconds) - now).total_seconds()
            if wait > 0:
                st.info(f"Cooling down for {wait:.1f}s to respect API limits…")
                time_module.sleep(wait)
        self.requests.append(datetime.now())

class KitAPI:
    def __init__(self, api_secret: str):
        self.headers = {
            "Accept": "application/json",
            "X-Kit-Api-Key": api_secret
        }
        self.limiter = RateLimiter(MAX_REQUESTS_PER_WINDOW, RATE_LIMIT_WINDOW)

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        url = f"{API_BASE_URL}/{endpoint}"
        retries = 0
        while True:
            self.limiter.wait_if_needed()
            try:
                resp = requests.request(method, url, headers=self.headers, params=params, timeout=30)
                if resp.status_code == 429:
                    retries += 1
                    if retries > MAX_RETRIES:
                        st.error("Hit rate limit too many times. Try narrower filters or wait a bit.")
                        return None
                    retry_after = resp.headers.get("Retry-After")
                    backoff = float(retry_after) if retry_after else min(60, 2 ** retries)
                    st.warning(f"Rate limited. Backing off {backoff:.1f}s…")
                    time_module.sleep(backoff)
                    continue
                resp.raise_for_status()
                # gentle pacing between successful requests to avoid bursts
                time_module.sleep(GENTLE_PACING_SEC)
                return resp
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > MAX_RETRIES:
                    st.error(f"API error: {e}")
                    return None
                backoff = min(30, 1.5 ** retries)
                st.warning(f"Network/API error. Retrying in {backoff:.1f}s…")
                time_module.sleep(backoff)

    def list_broadcasts(self, after: Optional[str] = None, per_page: int = DEFAULT_PER_PAGE) -> Dict:
        params = {"per_page": per_page}
        if after:
            params["after"] = after
        resp = self._request("GET", "broadcasts", params=params)
        return resp.json() if resp is not None else {}

    def get_broadcast(self, bid: int) -> Dict:
        resp = self._request("GET", f"broadcasts/{bid}")
        if not resp:
            return {}
        data = resp.json() or {}
        return data.get("broadcast", data)

    def get_broadcast_stats(self, bid: int) -> Dict:
        resp = self._request("GET", f"broadcasts/{bid}/stats")
        if not resp:
            return {}
        data = resp.json() or {}
        return data.get("broadcast", {}).get("stats", data.get("stats", {}))

# ──────────────────────────────────────────────────────────────────────────────
# Core download logic
# ──────────────────────────────────────────────────────────────────────────────
def download_broadcasts(api: KitAPI, start_iso: str, end_iso: str, published_filter: str, public_filter: str) -> pd.DataFrame:
    """
    Pages through broadcasts, pre-filters with lightweight list fields when possible,
    only enriches items that survive, and returns a tidy DataFrame.
    """
    start_dt = _parse_iso_z(start_iso)
    end_dt = _parse_iso_z(end_iso)

    rows: List[Dict] = []
    after = None
    scanned = kept = 0

    status = st.empty()
    prog = st.progress(0.0, text="Starting…")

    while True:
        page = api.list_broadcasts(after=after, per_page=DEFAULT_PER_PAGE)
        items = page.get("broadcasts", []) or []
        if not items:
            break

        for b in items:
            scanned += 1

            # Use list-level fields to pre-filter when available
            created_raw = b.get("created_at") or b.get("send_at")
            if start_dt and end_dt and not in_range(created_raw, start_dt, end_dt):
                # If list doesn't have created_at, we'll enrich to check properly
                if not created_raw:
                    details = api.get_broadcast(b["id"])
                    created_raw = details.get("created_at") or details.get("send_at")
                    if not in_range(created_raw, start_dt, end_dt):
                        prog.progress(min(0.99, kept / max(1, scanned)), text=f"Scanned {scanned}, kept {kept}…")
                        continue
                else:
                    prog.progress(min(0.99, kept / max(1, scanned)), text=f"Scanned {scanned}, kept {kept}…")
                    continue

            # quick filter check with whatever we have
            pre = {
                "send_at": b.get("send_at"),
                "public": b.get("public", False),
            }
            if not matches_filters(pre, published_filter, public_filter):
                # may need details if fields are missing/None
                details = api.get_broadcast(b["id"])
                pre["send_at"] = pre["send_at"] or details.get("send_at")
                pre["public"] = pre["public"] if pre["public"] is not None else details.get("public", False)
                if not matches_filters(pre, published_filter, public_filter):
                    prog.progress(min(0.99, kept / max(1, scanned)), text=f"Scanned {scanned}, kept {kept}…")
                    continue

            # Enrich only now (details + stats)
            details = api.get_broadcast(b["id"])
            stats = api.get_broadcast_stats(b["id"])
            combined = {**b, **(details or {}), **(stats or {})}
            rows.append(combined)
            kept += 1

            if scanned % 20 == 0:
                prog.progress(min(0.99, kept / max(1, scanned)), text=f"Scanned {scanned}, kept {kept}…")

        pag = page.get("pagination", {}) or {}
        if not pag.get("has_next_page"):
            break
        after = pag.get("end_cursor")

    prog.progress(1.0, text=f"Done. Scanned {scanned}, kept {kept}.")
    status.empty()
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.title("Kit Email Archive Tool")
    st.markdown(
        "<small>by <a href='https://omariharebin.com/tools' target='_blank'>Omari Harebin</a></small>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Download your Kit (formerly ConvertKit) broadcasts as **CSV** or **Markdown files** (one .md per email)."
    )

    # Session state for persistence across reruns
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if "zip_bytes" not in st.session_state:
        st.session_state.zip_bytes = b""

    with st.expander("🔑 Where to Find Your API Key", expanded=False):
        st.markdown(
            """
1. Log in to Kit  
2. Go to **Settings → API & Webhooks**  
3. Copy your **API Key** (keep it private)
"""
        )

    api_secret = st.text_input("API Secret", type="password", help="Find this in your Kit Account Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        # Default: very wide range
        date_range = st.date_input(
            "Date range",
            [datetime(2000, 1, 1).date(), datetime.now().date()],
        )
    with col2:
        published_filter = st.selectbox("Published status", ["All", "Published", "Draft"])
    with col3:
        public_filter = st.selectbox("Public status", ["All", "Public", "Private"])

    # Convert date range to inclusive ISO (end of day)
    if len(date_range) != 2:
        st.stop()

    start_iso = datetime.combine(date_range[0], time.min).isoformat() + "Z"
    end_iso = datetime.combine(date_range[1], time.max).isoformat() + "Z"

    left, right = st.columns([1, 1])

    with left:
        if st.button("Download Broadcasts", type="primary", use_container_width=True, disabled=not api_secret):
            st.session_state.df = pd.DataFrame()
            st.session_state.zip_bytes = b""
            api = KitAPI(api_secret)
            df = download_broadcasts(api, start_iso, end_iso, published_filter, public_filter)
            st.session_state.df = df

            if df.empty:
                st.warning("No broadcasts found for the given filters.")
            else:
                st.success(f"Loaded {len(df)} broadcasts.")

    if not st.session_state.df.empty:
        csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")

        with right:
            # Prepare ZIP only when first needed (and cache in session)
            if st.session_state.zip_bytes == b"":
                st.session_state.zip_bytes = save_to_markdown_zip(st.session_state.df)

            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="kit_broadcasts.csv",
                mime="text/csv",
                use_container_width=True,
                key="csv_dl",
            )
            st.download_button(
                "Download Markdown ZIP",
                data=st.session_state.zip_bytes,
                file_name="kit_broadcasts_markdown.zip",
                mime="application/zip",
                use_container_width=True,
                key="zip_dl",
            )

        st.divider()
        st.dataframe(st.session_state.df, use_container_width=True, height=480)
    else:
        with right:
            st.button("Download CSV", disabled=True, use_container_width=True)
            st.button("Download Markdown ZIP", disabled=True, use_container_width=True)

if __name__ == "__main__":
    main()
