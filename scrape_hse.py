import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup, Tag, NavigableString
from urllib.parse import urljoin, urlparse

BASE_URL = "https://www2.hse.ie"
INDEX_URL = "https://www2.hse.ie/conditions/"
OUTPUT_CSV = "hse_conditions_long.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HSE-Medical-Scraper/1.0)"
}

REQUEST_DELAY = 1.0

session = requests.Session()
session.headers.update(HEADERS)


# ----------------------------
# text helpers
# ----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace("’", "").replace("'", "")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def clean_filename(name: str) -> str:
    name = clean_text(name).lower()
    name = name.replace("’", "").replace("'", "")
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")


# ----------------------------
# heading normalization
# ----------------------------
DROP_H2_HEADINGS = {
    "contents",
    "support links",
    "hse live - we're here to help",
}

STOP_H2_HEADINGS = {
    "support links",
    "hse live - we're here to help",
}

SECTION_MAP_EXACT = {
    "abdominal_aortic_aneurysm_and_driving": "driving",
}

SECTION_MAP_PREFIX = {
    "symptoms_of": "symptoms",
    "tests_for": "tests",
    "treatment_for": "treatment",
    "reducing_risks_of": "reducing_risks",
    "whos_at_risk_of": "risk_factors",
    "whos_at_risk": "risk_factors",
    "causes_of": "causes",
    "preventing": "prevention",
    "prevention_of": "prevention",
    "living_with": "living_with",
}


def normalize_heading(heading: str, condition_name: str) -> str:
    text = clean_text(heading).lower()
    text = text.replace("’", "").replace("'", "")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)

    if text in SECTION_MAP_EXACT:
        return SECTION_MAP_EXACT[text]

    for prefix, normalized in SECTION_MAP_PREFIX.items():
        if text.startswith(prefix):
            return normalized

    condition_slug = clean_filename(condition_name)
    if text == f"{condition_slug}_and_driving":
        return "driving"

    return text


# ----------------------------
# html extraction helpers
# ----------------------------
def get_soup(url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_condition_links() -> list[str]:
    soup = get_soup(INDEX_URL)
    links = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(BASE_URL, href)

        if "/conditions/" not in full_url:
            continue

        if full_url.rstrip("/") == INDEX_URL.rstrip("/"):
            continue

        path = full_url.replace(BASE_URL, "")
        if re.match(r"^/conditions/[^/]+/?$", path):
            if full_url not in seen:
                seen.add(full_url)
                links.append(full_url)

    return sorted(links)


def extract_condition_name(main: Tag) -> str:
    h1 = main.find("h1")
    if not h1:
        return ""

    title = clean_text(h1.get_text(" ", strip=True))
    if " - " in title:
        left, right = title.split(" - ", 1)
        if left.strip().lower() == "overview":
            return right.strip()
    return title


def is_meaningful_text(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False

    junk = {
        "contents",
        "overview",
        "treatment",
        "back to top",
    }
    if text.lower() in junk:
        return False

    return True


def format_list(tag: Tag) -> str:
    items = []
    for li in tag.find_all("li", recursive=False):
        item_text = clean_text(li.get_text(" ", strip=True))
        if item_text:
            items.append(f"- {item_text}")
    return "\n".join(items).strip()


def format_callout(tag: Tag) -> str:
    """
    Handles things like:
    ### Non-urgent advice: Contact your GP if you:
    ### Emergency action required: Call 999 or 112 ...
    """
    title = clean_text(tag.get_text(" ", strip=True))
    if not title:
        return ""

    parts = [title]

    for sibling in tag.next_siblings:
        if isinstance(sibling, Tag) and sibling.name in {"h2", "h3"}:
            break

        if isinstance(sibling, Tag):
            if sibling.name in {"p", "div"}:
                txt = clean_text(sibling.get_text(" ", strip=True))
                if txt:
                    parts.append(txt)
            elif sibling.name in {"ul", "ol"}:
                list_text = format_list(sibling)
                if list_text:
                    parts.append(list_text)

    return "\n".join(parts).strip()


def extract_overview(main: Tag, first_h2: Tag) -> str:
    content_parts = []
    current = first_h2.previous_sibling

    # walk backwards until h1, then reverse
    tmp = []
    while current:
        if isinstance(current, Tag) and current.name == "h1":
            break
        if isinstance(current, Tag):
            tmp.append(current)
        current = current.previous_sibling

    for node in reversed(tmp):
        if not isinstance(node, Tag):
            continue

        if node.name in {"ul", "ol", "nav"}:
            text = clean_text(node.get_text(" ", strip=True))
            # skip the Contents TOC
            if text.lower().startswith("overview") or text.lower().startswith("contents"):
                continue
            list_text = format_list(node)
            if list_text:
                content_parts.append(list_text)

        elif node.name in {"p", "div"}:
            text = clean_text(node.get_text(" ", strip=True))
            if is_meaningful_text(text):
                content_parts.append(text)

    overview = "\n\n".join(part for part in content_parts if part).strip()
    return overview


def extract_section_content(h2: Tag) -> str:
    parts = []

    for sibling in h2.next_siblings:
        if isinstance(sibling, Tag) and sibling.name == "h2":
            break

        if isinstance(sibling, NavigableString):
            continue

        if not isinstance(sibling, Tag):
            continue

        if sibling.name == "p":
            txt = clean_text(sibling.get_text(" ", strip=True))
            if txt:
                parts.append(txt)

        elif sibling.name in {"ul", "ol"}:
            list_text = format_list(sibling)
            if list_text:
                parts.append(list_text)

        elif sibling.name == "h3":
            callout_text = format_callout(sibling)
            if callout_text:
                parts.append(callout_text)

        elif sibling.name == "div":
            txt = clean_text(sibling.get_text(" ", strip=True))
            if txt and len(txt) > 20:
                parts.append(txt)

    content = "\n\n".join(part for part in parts if part).strip()
    return content


def scrape_condition(url: str) -> list[dict]:
    soup = get_soup(url)
    main = soup.find("main") or soup

    condition_name = extract_condition_name(main)
    source_path = urlparse(url).path
    rows = []

    h2_tags = main.find_all("h2")
    if not h2_tags:
        return rows

    # overview
    first_h2 = h2_tags[0]
    if clean_text(first_h2.get_text(" ", strip=True)).lower() == "contents":
        overview = extract_overview(main, first_h2)
        if overview:
            rows.append({
                "condition_name": condition_name,
                "section": "overview",
                "content": overview,
                "source_url": url,
                "source_path": source_path,
            })

    # sections
    for h2 in h2_tags:
        raw_heading = clean_text(h2.get_text(" ", strip=True))
        if not raw_heading:
            continue

        raw_heading_lower = raw_heading.lower()
        if raw_heading_lower in DROP_H2_HEADINGS:
            continue
        if raw_heading_lower in STOP_H2_HEADINGS:
            break

        section = normalize_heading(raw_heading, condition_name)
        content = extract_section_content(h2)

        if not content or len(clean_text(content)) < 30:
            continue

        anchor = slugify(raw_heading)
        rows.append({
            "condition_name": condition_name,
            "section": section,
            "content": content,
            "source_url": f"{url}#{anchor}",
            "source_path": source_path,
        })

    return rows


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop_duplicates(subset=["condition_name", "section", "content"])
          .reset_index(drop=True)
    )


def main():
    print("Collecting condition links...")
    links = extract_condition_links()
    print(f"Found {len(links)} condition links")

    all_rows = []

    for i, url in enumerate(links, start=1):
        try:
            print(f"[{i}/{len(links)}] Scraping {url}")
            rows = scrape_condition(url)
            all_rows.extend(rows)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"Failed: {url} -> {e}")

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No data scraped.")
        return

    df = remove_duplicates(df)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Rows: {len(df)}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()