import redis.asyncio as redis
import json
from config import redis_url

class MatchCache:
    def __init__(self, ttl=43200):
        self.redis = None
        self.ttl = ttl

    async def start(self):
        self.redis = await redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=10)


    async def save_match(self, player_a: str, player_b: str, odds_a: float, odds_b: float):
        key = f'{player_a}:{player_b}'
        value = json.dumps({
            'odds_a': odds_a,
            'odds_b': odds_b,
        })
        await self.redis.set(key, value, ex=self.ttl)

    async def get_all_matches(self):
        matches = {}
        for key in await self.redis.keys('*:*'):
            value = await self.redis.get(key)
            if not value:
                continue
            value = json.loads(value)
            matches[tuple(key.split(':'))] = (value['odds_a'], value['odds_b'])

        return matches

    async def get_odds_for_match(self, player_a: str, player_b: str):
        value = await self.redis.get(f'{player_a}:{player_b}') or await self.redis.get(f'{player_b}:{player_a}')
        if not value:
            return None, None
        value = json.loads(value)
        if await self.redis.exists(f'{player_a}:{player_b}'):
            return value['odds_a'], value['odds_b']
        return value['odds_b'], value['odds_a']

    async def ping(self):
        return await self.redis.ping()