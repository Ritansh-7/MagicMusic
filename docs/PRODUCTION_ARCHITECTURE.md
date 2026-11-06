# CozyBeats production architecture

Docker is optional. Local development uses PowerShell, the existing Python virtual environment, and Node.js.

```text
Browser / mobile web
        |
        | HTTPS
        v
Next.js frontend (Vercel or a Windows/Linux Node host)
        |
        | REST/JSON
        v
FastAPI replicas (2+ instances behind a load balancer)
        |                         |
        |                         +--> Redis / Upstash (cache, rate limits)
        +--> PostgreSQL / Supabase (users, events, playlists)
        |
        +--> JioSaavn adapter + local recommendation catalog

Streamlit app: research, model evaluation, admin analytics, internal prototype
```

## Windows development without Docker

Terminal 1:

```powershell
streamlit run app.py
```

Terminal 2:

```powershell
.\start_api.ps1
```

Terminal 3:

```powershell
.\start_frontend.ps1
```

Open `http://localhost:3000`. The API health check is `http://localhost:8000/healthz`.

Redis and PostgreSQL are optional locally. Without `REDIS_URL`, the API uses a bounded in-memory TTL cache. Without a database URL, API events use SQLite. For production, use managed Redis and PostgreSQL so multiple API replicas share state.

## 500-user baseline

- Keep API replicas stateless.
- Use 2 API instances with 1–2 Uvicorn workers each to start; scale horizontally after load testing.
- Put Redis in front of provider searches and recommendation responses.
- Add a per-user/IP rate limit before public launch: search 30/minute, recommendations 20/minute, events 120/minute.
- Keep JioSaavn calls bounded and cached; never call the provider directly from the browser.
- Use PostgreSQL connection pooling and indexes on `(user_id, created_at)` and `(user_id, track_id)`.
- Serve artwork/audio through a CDN or provider URLs; do not proxy large audio through FastAPI.
- Add health checks, structured logs, request IDs, timeouts, retries with backoff, and provider circuit breaking.
- Load-test with 500 virtual users before claiming production capacity.

## Security before public launch

- Add OAuth/password auth and signed access tokens.
- Validate ownership for every playlist and event.
- Restrict CORS to the deployed frontend origin.
- Keep database and Redis URLs in hosting secrets.
- Add request-size limits, abuse throttling, and account deletion/export flows.
- Confirm audio licensing and provider terms before commercial use.
