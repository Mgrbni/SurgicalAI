import time, random
import httpx

def retry_policy(fn, max_tries=3, base=0.5, cap=4.0):
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except (httpx.ReadTimeout, httpx.ConnectError):
            if attempt == max_tries:
                raise
        except Exception as e:
            # If OpenAI 429/5xx, backoff; else raise immediately
            msg = str(e).lower()
            if ("429" not in msg and "rate" not in msg and "overload" not in msg and "5" not in msg):
                raise
        sleep = min(cap, base * (2 ** (attempt - 1))) + random.random() * 0.25
        time.sleep(sleep)
