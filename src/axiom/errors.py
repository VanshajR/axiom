"""What can go wrong with a model call, sorted by what the router should do about it."""


class ProviderError(Exception):
    """A model call failed. `retryable` tells the router to try the next model."""

    retryable = False

    def __init__(self, provider: str, model: str, message: str, status: int | None = None):
        super().__init__(f"{provider}/{model}: {message}")
        self.provider, self.model, self.status = provider, model, status


class RateLimited(ProviderError):
    retryable = True


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


def classify(provider: str, model: str, status: int, body: str) -> ProviderError:
    text = body[:300]
    lowered = body.lower()
    if status == 429:
        return RateLimited(provider, model, text, status)
    if status in (401, 403):
        return AuthFailed(provider, model, text, status)
    if status == 404 or "decommissioned" in lowered or "no longer available" in lowered or "model_not_found" in lowered:
        return ModelUnavailable(provider, model, text, status)
    if status >= 500:
        return Overloaded(provider, model, text, status)
    return BadRequest(provider, model, text, status)
