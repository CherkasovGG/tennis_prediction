import redis
import json
from config import redis_url

class MatchCache:
    def __init__(self, ttl=43200):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl

    def save_match(self, player_a: str, player_b: str, odds_a: float, odds_b: float):
        key = f'{player_a}:{player_b}'
        value = json.dumps({
            'odds_a': odds_a,
            'odds_b': odds_b,
        })
        self.redis.set(key, value, ex=self.ttl)

    def get_all_matches(self):
        matches = {}
        for key in self.redis.keys('*:*'):
            value = self.redis.get(key)
            if not value:
                continue
            value = json.loads(value)
            matches[tuple(key.split(':'))] = (value['odds_a'], value['odds_b'])

        return matches

    def get_odds_for_match(self, player_a: str, player_b: str):
        value = self.redis.get(f'{player_a}:{player_b}') or self.redis.get(f'{player_b}:{player_a}')
        if not value:
            return None, None
        value = json.loads(value)
        if self.redis.exists(f'{player_a}:{player_b}'):
            return value['odds_a'], value['odds_b']
        return value['odds_b'], value['odds_a']

    def ping(self):
        return self.redis.ping()