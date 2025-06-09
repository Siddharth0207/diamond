import redis  # Import the redis-py client
import os
import json
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# Get the Redis connection URL from environment or use default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# docker run -d --name redis-local -p 6379:6379 redis  # Example command to run Redis in Docker

class RedisSessionManager:
    def __init__(self, url=REDIS_URL):
        """
        Initialize the RedisSessionManager with a Redis client.
        Args:
            url (str): The Redis connection URL.
        """
        # Initialize the Redis client using the provided URL
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def set_session(self, session_id: str, data: dict, expire: int = 3600):
        """
        Store session data in Redis as a JSON string with an expiration time.
        Args:
            session_id (str): Unique identifier for the session.
            data (dict): Session data to store.
            expire (int): Expiration time in seconds (default: 3600).
        """
        # Store session data as a JSON string with an expiration time (default 1 hour)
        self.client.set(session_id, json.dumps(data), ex=expire)

    def get_session(self, session_id: str) -> dict:
        """
        Retrieve session data from Redis and parse it from JSON.
        Args:
            session_id (str): Unique identifier for the session.
        Returns:
            dict: The session data, or an empty dict if not found.
        """
        # Retrieve session data by session_id and parse from JSON
        data = self.client.get(session_id)
        if data:
            return json.loads(data)
        return {}

    def update_session(self, session_id: str, updates: dict):
        """
        Update existing session data in Redis with new values.
        Args:
            session_id (str): Unique identifier for the session.
            updates (dict): Dictionary of updates to apply to the session.
        """
        # Update existing session data with new values
        session = self.get_session(session_id)
        session.update(updates)
        self.set_session(session_id, session)

    def delete_session(self, session_id: str):
        """
        Delete the session data for the given session_id from Redis.
        Args:
            session_id (str): Unique identifier for the session.
        """
        # Delete the session data for the given session_id
        self.client.delete(session_id)
