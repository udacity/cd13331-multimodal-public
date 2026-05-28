import asyncio
import time
from functools import wraps

import asyncio
import time
from functools import wraps

def _get_rate_limiter(calls: int, period: int):
    """
    Create a rate limiting context that can be shared between decorators.
    Returns a coroutine that handles the rate limiting logic.
    
    Args:
        calls (int): Maximum number of calls allowed within the period.
        period (int): Time window in seconds.
    """
    interval = period / calls
    next_call = [time.time()]
    lock = asyncio.Lock()
    
    async def apply_rate_limit():
        """Apply rate limiting by sleeping if necessary."""
        # Quickly assign a time slot
        async with lock:
            run_at = max(next_call[0], time.time())
            next_call[0] = run_at + interval
        
        # Sleep outside the lock (non-blocking for other requests)
        wait_time = run_at - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
    
    return apply_rate_limit


def rate_limit(calls: int, period: int):
    """
    Decorate a regular async function to limit it to `calls` within `period` seconds.
    This guarantees a uniform distribution of calls over time, without exceeding the rate limit.

    NOTE: this implementation is async safe but not process safe nor thread safe.
    """
    apply_rate_limit = _get_rate_limiter(calls, period)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await apply_rate_limit()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit_streaming(calls: int, period: int):
    """
    Decorate an async generator (streaming function) to limit it to `calls` within `period` seconds.
    This guarantees a uniform distribution of calls over time, without exceeding the rate limit.

    NOTE: this implementation is async safe but not process safe nor thread safe.
    """
    apply_rate_limit = _get_rate_limiter(calls, period)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await apply_rate_limit()
            async for item in func(*args, **kwargs):
                yield item
        return wrapper
    return decorator