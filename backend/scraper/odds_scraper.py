import asyncio
import json
import html
from urllib.parse import urljoin
from playwright.async_api import async_playwright
from backend.config import base_url, odds_url

class OddsScraper:
    def __init__(self):
        self.playwright = None
        self.browser = None

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"]
        )

    async def close(self):
        await self.browser.close()
        await self.playwright.stop()


    async def find_match_odds(self):
        match_odds = {}

        context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US"
        )
        try:
            atp_match_groups_links = await self._extract_links(context)
            for group_link in atp_match_groups_links or []:
                print(group_link)
                matches = await self._extract_links_from_ldjson(group_link, context)
                for names, link in matches or []:
                    try:
                        odds = await asyncio.wait_for(self._get_odds(link, context), timeout=15)
                        if odds:
                            match_odds[names] = odds
                            print(names, odds)
                    except Exception as e:
                        print(f"[ERROR get_odds] {names} - {e}")
            print('Odds finding ended')
        except Exception as e:
            print(f"[ERROR find_match_odds] {e}")
        finally:
            if context:
                await context.close()
        return match_odds

    def _normalize_link(self, href: str) -> str:
        if not href:
            return None
        return urljoin(base_url, href)

    async def _get_odds(self, url, context):
        print("Started getting odds for", url)
        page = await context.new_page()
        try:
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(6000)
            max_scrolls = 20
            scroll_count = 0
            last_height = 0
            while scroll_count < max_scrolls:
                print(f"Scroll {scroll_count} / {max_scrolls} for {url}")
                height = await page.evaluate("document.body.scrollHeight")
                if height == last_height:
                    break
                last_height = height
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)
                scroll_count += 1
            await page.wait_for_timeout(3000)
            rows = await page.query_selector_all("div[data-testid*='expanded-row']")
            odds_data = {}
            for row in rows:
                try:
                    text = await row.inner_text()
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
            if "1xBet" in odds_data:
                return odds_data["1xBet"]
            if odds_data:
                return next(iter(odds_data.values()))
        except Exception as e:
            print(f"[ERROR get_odds] {e}")
        finally:
            await page.close()
        return None

    async def _extract_links_from_ldjson(self, url, context):
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)
            scripts = page.locator("script[type='application/ld+json']")
            count = await scripts.count()
            links = []
            for i in range(count):
                content = await scripts.nth(i).inner_text()
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
                            names = tuple(html.unescape(item.get("name")).split(' - '))
                            match_url = item.get("url")
                            links.append((names, match_url))
            return links
        except Exception as e:
            print(f"[ERROR extract_links_from_ldjson] {e}")
            return []
        finally:
            await page.close()

    async def _extract_links(self, context):
        page = await context.new_page()
        try:
            await page.goto(odds_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(5000)
            selector = """#app > div.relative.flex.flex-col.w-full.max-w-\\[1350px\\].font-main > 
            div.w-full.flex-center.bg-gray-med_light > 
            div > main > 
            div.relative.w-full.flex-grow-1.min-w-\\[320px\\].bg-white-main > 
            div:nth-child(3)"""
            container = page.locator(selector)
            if await container.count() == 0:
                print("[ERROR extract_links] Блок не найден")
                return []
            links = container.locator("a")
            count = await links.count()
            normalized_links = []
            for i in range(count):
                text = await links.nth(i).inner_text()
                if 'atp' in text.lower() and 'women' not in text.lower() and 'doubles' not in text.lower():
                    href = await links.nth(i).get_attribute("href")
                    full_url = self._normalize_link(href)
                    normalized_links.append(full_url)
            return normalized_links
        except Exception as e:
            print(f"[ERROR extract_links] {e}")
            return []
        finally:
            await page.close()