"""What can go wrong with a model call, sorted by what the router should do about it."""
import re


class ProviderError(Exception):
    """A model call failed. `retryable` tells the router to try the next model."""

    retryable = False

    def __init__(self, provider: str, model: str, message: str, status: int | None = None):
        super().__init__(f"{provider}/{model}: {message}")
        self.provider, self.model, self.status = provider, model, status


class RateLimited(ProviderError):
    """The provider says how long to wait (Groq: "try again in 472ms", Gemini: a
    retryDelay). Per-minute token limits refill continuously, so that's often
    under a second, and the router waits it out instead of moving on."""

    retryable = True

    def __init__(self, provider: str, model: str, message: str, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(provider, model, message, status)
        self.retry_after = retry_after


class ModelUnavailable(ProviderError):
    """Retired, renamed, or "no longer available to new users". Skip it for a long while."""

    retryable = True


class Overloaded(ProviderError):
    """503s, timeouts, dropped connections: usually gone in a minute."""

    retryable = True


class AuthFailed(ProviderError):
    """Bad or missing key. Retrying the same provider won't help, another one might."""

    retryable = True


class BadRequest(ProviderError):
    """A 400. Sometimes it's model-specific (Gemini's thought_signature rule), so
    another model gets a try, but the model isn't benched for it."""

    retryable = True


_UNITS = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}


def retry_after(body: str, headers=None) -> float | None:
    """Seconds the provider asked us to wait, if it said."""
    header = (headers or {}).get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    # Groq: "Please try again in 472.5ms" / "in 1m26.4s"; Gemini: "retryDelay": "37s"
    match = re.search(r'(?:try again in|retryDelay"?:\s*"?)\s*((?:[\d.]+(?:ms|h|m|s))+)', body)
    if not match:
        return None
    return sum(float(n) * _UNITS[unit] for n, unit in re.findall(r"([\d.]+)(ms|h|m|s)", match.group(1)))


def classify(provider: str, model: str, status: int, body: str, headers=None) -> ProviderError:
    text = body[:300]
    lowered = body.lower()
    if status == 429:
        return RateLimited(provider, model, text, status, retry_after(body, headers))
    if status in (401, 403):
        return AuthFailed(provider, model, text, status)
    if status == 404 or "decommissioned" in lowered or "no longer available" in lowered or "model_not_found" in lowered:
        return ModelUnavailable(provider, model, text, status)
    if status >= 500:
        return Overloaded(provider, model, text, status)
    return BadRequest(provider, model, text, status)
