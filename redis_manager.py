import redis
import os
import json

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class RedisSessionManager:
    def __init__(self, url=REDIS_URL):
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def set_session(self, session_id: str, data: dict, expire: int = 3600):
        self.client.set(session_id, json.dumps(data), ex=expire)

    def get_session(self, session_id: str) -> dict:
        data = self.client.get(session_id)
        if data:
            return json.loads(data)
        return {}

    def update_session(self, session_id: str, updates: dict):
        session = self.get_session(session_id)
        session.update(updates)
        self.set_session(session_id, session)

    def delete_session(self, session_id: str):
        self.client.delete(session_id)
