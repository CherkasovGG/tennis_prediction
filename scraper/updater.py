import time

class OddsUpdater:
    def __init__(self, scrapper, match_cache, update_interval=900):
        self.scrapper = scrapper
        self.match_cache = match_cache
        self.update_interval = update_interval

    def start(self):
        while True:
            try:
                data = self.scrapper.find_match_odds()

                for (p1, p2), (odds_a, odds_b) in data.items():
                    print("SAVING TO REDIS:", p1, p2)
                    self.match_cache.save_match(p1, p2, odds_a, odds_b)

            except Exception as e:
                print(e)

            time.sleep(self.update_interval)