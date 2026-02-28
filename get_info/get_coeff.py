from collections import defaultdict

from config.config import base_url, odds_url
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
import json
import time

def normalize_link(href: str) -> str:
    if not href:
        return None
    return urljoin(base_url, href)

def find_match_odds():
    match_odds = defaultdict(dict)
    atp_match_groups_links = extract_links()
    for group_link in atp_match_groups_links:
        matches = extract_links_from_ldjson(group_link)
        for name, link in matches:
            odds = get_odds(link)
            match_odds[name] = odds

    return match_odds


def get_odds(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US"
        )

        page = context.new_page()

        # Убираем navigator.webdriver
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(6000)

        last_height = 0
        while True:
            height = page.evaluate("document.body.scrollHeight")
            if height == last_height:
                break
            last_height = height
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)

        page.wait_for_timeout(3000)

        rows = page.query_selector_all("div[data-testid*='expanded-row']")

        odds_data = {}

        for row in rows:
            try:
                text = row.inner_text()
                lines = text.split("\n")

                if len(lines) >= 3:
                    bookmaker = lines[0].strip()

                    numbers = []
                    for line in lines[1:]:
                        try:
                            num = float(line.strip())
                            numbers.append(num)
                        except:
                            continue

                    if len(numbers) >= 2:
                        odds_data[bookmaker] = numbers[:2]

            except:
                continue

        browser.close()
        return odds_data

def extract_links_from_ldjson(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_timeout(3000)

        scripts = page.locator("script[type='application/ld+json']")
        count = scripts.count()
        links = []

        for i in range(count):
            content = scripts.nth(i).inner_text()
            try:
                data = json.loads(content)
            except:
                continue

            items = data if isinstance(data, list) else [data]

            for item in items:
                types = item.get("@type")
                if types:
                    if isinstance(types, str):
                        types = [types]
                    if "Event" in types and item.get("url"):
                        name = item.get("name")
                        match_url = item.get("url")
                        links.append((name, match_url))

        browser.close()
        return links

def extract_links():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(odds_url, timeout=60000)
        page.wait_for_timeout(5000)

        selector = """#app > div.relative.flex.flex-col.w-full.max-w-\\[1350px\\].font-main > 
        div.w-full.flex-center.bg-gray-med_light > 
        div > main > 
        div.relative.w-full.flex-grow-1.min-w-\\[320px\\].bg-white-main > 
        div:nth-child(3)"""

        container = page.locator(selector)
        if container.count() == 0:
            print("Блок не найден")
            browser.close()
            return

        links = container.locator("a")
        count = links.count()

        normalized_links = []

        for i in range(count):
            text = links.nth(i).inner_text()
            if 'atp' in text.lower() and 'women' not in text.lower():
                href = links.nth(i).get_attribute("href")
                full_url = normalize_link(href)

                normalized_links.append(full_url)

        browser.close()
        return normalized_links

if __name__ == "__main__":
    print(find_match_odds())