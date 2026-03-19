import asyncio

class OddsUpdater:
    def __init__(self, scrapper, match_cache, update_interval=900):
        self.scrapper = scrapper
        self.match_cache = match_cache
        self.update_interval = update_interval

    async def start(self):
        while True:
            try:
                data = await self.scrapper.find_match_odds()

                for (p1, p2), (odds_a, odds_b) in data.items():
                    print("SAVING TO REDIS:", p1, p2)
                    await self.match_cache.save_match(p1, p2, odds_a, odds_b)

                print("REDIS KEYS AFTER SAVE:", await self.match_cache.redis.keys("*"))

            except Exception as e:
                print(e)

            await asyncio.sleep(self.update_interval)