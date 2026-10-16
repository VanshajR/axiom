# Hugging Face Docker Space: runs as uid 1000 and expects port 7860
FROM python:3.12-slim

RUN useradd -m -u 1000 app
WORKDIR /home/app/axiom

COPY --chown=app pyproject.toml README.md ./
COPY --chown=app src ./src
RUN pip install --no-cache-dir .
# fallback copy of the site's data, refreshed by scripts/sync_snapshot.sh
COPY --chown=app profile.snapshot.json ./

USER app
EXPOSE 7860
CMD ["uvicorn", "axiom.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
