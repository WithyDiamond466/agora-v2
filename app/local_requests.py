"""Allow the local app and CLI, reject unrelated browser origins and hosts."""
from urllib.parse import urlsplit
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class LocalRequestsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        host = request.url.hostname
        test_client = request.client and request.client.host == 'testclient' and host == 'testserver'
        if host not in {'localhost', '127.0.0.1', '::1'} and not test_client:
            return JSONResponse({'detail': 'Agora accepts local hostnames only.'}, status_code=400)
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('origin')
            source = origin or request.headers.get('referer')
            if source:
                try:
                    parsed = urlsplit(source)
                    source_port = parsed.port or (443 if parsed.scheme == 'https' else 80)
                except ValueError:
                    return JSONResponse({'detail': 'Invalid request origin.'}, status_code=403)
                target_port = request.url.port or (443 if request.url.scheme == 'https' else 80)
                if (parsed.scheme, parsed.hostname, source_port) != (request.url.scheme, host, target_port):
                    return JSONResponse({'detail': 'Cross-origin changes are not allowed.'}, status_code=403)
            elif request.headers.get('sec-fetch-site') in {'cross-site', 'same-site'}:
                return JSONResponse({'detail': 'Cross-origin changes are not allowed.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "frame-ancestors 'none'"
        response.headers['Referrer-Policy'] = 'same-origin'
        if not request.url.path.startswith('/static/'):
            response.headers['Cache-Control'] = 'no-store'
        return response
