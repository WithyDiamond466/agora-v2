"""FastAPI application factory and wiring."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import config
from app.db import init_db
from app.local_requests import LocalRequestsMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agora")

ROUTER_MODULES = [
    "app.routers.courses",
    "app.routers.grading",
    "app.routers.skills",
    "app.routers.settings",
    "app.routers.chat",
    "app.routers.analytics",
    "app.routers.insight",  # Increment 1 · insight module
    "app.routers.terms",  # Fall readiness · terms of use
    "app.routers.release",  # Fall readiness · review record + release + export
    "app.routers.compare",  # Fall readiness · model comparison on real submissions
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("Agora ready at %s", config.base_url())
    yield
    log.info("Agora shutting down")


app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION, lifespan=lifespan)

app.add_middleware(LocalRequestsMiddleware)

config.ensure_dirs()
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

#: Which routers actually loaded — surfaced on /api/health for debugging.
LOADED_ROUTERS: list[str] = []


def _include_routers() -> None:
    for module_path in ROUTER_MODULES:
        module = importlib.import_module(module_path)
        router = getattr(module, "router", None)
        if router is None:
            raise RuntimeError(f"Router module {module_path} has no `router` attribute")
        app.include_router(router)
        LOADED_ROUTERS.append(module_path)
        log.info("Mounted router %s", module_path)


_include_routers()


# --------------------------------------------------------------------------
# lifecycle / health
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "routers": LOADED_ROUTERS,
        "missing_routers": [m for m in ROUTER_MODULES if m not in LOADED_ROUTERS],
        "database": str(config.DB_PATH),
    }


async def _stop_server(delay: float = 0.3) -> None:
    """Let the HTTP response flush, then stop uvicorn for real.

    Preferred path: flip `should_exit` on the uvicorn Server object that run.py
    parked on `app.state`. Fallback: SIGINT ourselves, which uvicorn's own
    signal handler turns into a graceful shutdown. Either way the process
    actually exits — no zombie server holding port 8811 (v1's classic bug).
    """
    await asyncio.sleep(delay)
    server = getattr(app.state, "server", None)
    if server is not None:
        server.should_exit = True
        return
    os.kill(os.getpid(), signal.SIGINT)


@app.post("/api/shutdown")
async def shutdown() -> JSONResponse:
    """Cleanly stop the local server (the browser tab's Quit button)."""
    log.info("Shutdown requested")
    asyncio.create_task(_stop_server())
    return JSONResponse({"status": "shutting_down"})


@app.exception_handler(404)
async def not_found(request: Request, exc: Any) -> Any:
    """HTML 404 for page routes, JSON for the API."""
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)
    try:
        return templates.TemplateResponse(request, "404.html", {}, status_code=404)
    except Exception:
        return HTMLResponse(
            "<h1>404 — page not found</h1><p><a href='/'>Back to Agora</a></p>",
            status_code=404,
        )


__all__ = ["app", "templates", "LOADED_ROUTERS"]
