from dotenv import load_dotenv
load_dotenv()

import os
from redis import Redis
from rq import Queue
from rq.worker import SimpleWorker

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
conn = Redis.from_url(redis_url)

if __name__ == "__main__":
    q = Queue("meetings", connection=conn)
    worker = SimpleWorker([q], connection=conn)
    worker.work()
