# For any Docker host (Render, Koyeb, a VPS). The live deployment is a Vercel function; see scripts/deploy_vercel.py.
FROM python:3.12-slim

RUN useradd -m -u 1000 app
WORKDIR /home/app/axiom

COPY --chown=app pyproject.toml README.md ./
COPY --chown=app src ./src
RUN pip install --no-cache-dir .
# fallback copy of the site's data, refreshed by scripts/sync_snapshot.sh
COPY --chown=app profile.snapshot.json ./

USER app
ENV PORT=8000
EXPOSE 8000
CMD uvicorn axiom.app:app --host 0.0.0.0 --port "$PORT" --proxy-headers --forwarded-allow-ips "*"
