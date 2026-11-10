# CozyBeats auth and API deployment

The web client uses FastAPI for registration, login, song shelves, search, and history. The Render configuration now defines `cozybeats-api` separately from the Streamlit research app.

## Required deployment settings

1. Deploy the `cozybeats-api` service from this repository's `render.yaml`. Its health check is `/healthz`; its start command runs `uvicorn backend.main:app` on Render's assigned port.
2. In Render, set the API service's `DATABASE_URL` to a reachable, persistent MySQL or PostgreSQL database URL. `UserStore` can use local SQLite for development, but a Render instance's local filesystem is not a reliable shared account database. Do not treat signup as durable until the managed database is configured and reachable.
3. Keep the generated `COZYBEATS_SESSION_SECRET` secret stable across API deploys/instances. Changing it invalidates existing login tokens.
4. In the Vercel project for `cozybeats.vercel.app`, set `NEXT_PUBLIC_API_URL` to the deployed API origin (for example, `https://cozybeats-api.onrender.com`, if that is the Render service's assigned hostname), with no `/v1` suffix. Set it for the Production environment and redeploy, since Next.js embeds this public variable in the frontend build.
5. If the Vercel domain changes, update the API service's `COZYBEATS_ALLOWED_ORIGINS` to the exact frontend origin and redeploy the API.

## Verify the flow

- Open `https://<your-api-host>/healthz`; expect JSON with `"ok": true`.
- Register once in the website. A successful response stores a signed token; the page immediately loads `/v1/home` and `/v1/history`.
- Log out and sign in with the same account. If registration reports a database error, inspect the Render API logs and verify the database URL/network access; do not retry signup repeatedly before checking whether the initial insert committed.

The frontend now disables repeat auth submissions while a request is in flight, displays API/network errors, and shows a song-loading state after successful auth. Production builds intentionally do not fall back to `localhost:8000`; a missing `NEXT_PUBLIC_API_URL` is shown as a configuration error instead.
