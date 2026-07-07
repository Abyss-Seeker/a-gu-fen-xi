# cloud-functions/api/[[default]].py
# ============================================================================
# EdgeOne Pages entry point for the stock deep-diagnostic tool.
#
# How EdgeOne maps this file:
#   cloud-functions/api/[[default]].py  ->  https://<project>.pages.dnsoe9.com/api/*
#   (the [[default]] catch-all matches every path segment under /api)
#
# IMPORTANT routing note:
#   EdgeOne strips the "/api" filesystem prefix BEFORE handing the request to
#   this WSGI app. So a browser request to  /api/analyze  arrives here as the
#   internal path  /analyze . Therefore every internal route below is
#   registered WITHOUT the "/api" prefix.
#
# Reuse strategy (zero logic duplication):
#   The full Flask app lives in app.py at the REPO ROOT (untouched, identical
#   to the Vercel/main branch). We import that app object purely to borrow its
#   already-defined view functions, then re-register them on a fresh Flask
#   instance with the "/api" prefix stripped. The root "/" page and the
#   "/static/<path>" routes are intentionally dropped, because EdgeOne serves
#   the static index.html and /static/* assets directly from the repo root.
# ============================================================================

import sys
import os
import re

# Make the repository root importable so we can reuse app.py / helpers.
# cloud-functions/api/[[default]].py -> 3 levels up = repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flask import Flask

# Import the original Flask app ONLY to reuse its view functions.
# We never use source_app for routing (its routes keep the "/api" prefix and
# its "/" route renders a Jinja2 template that does not exist on EdgeOne).
from app import app as source_app

app = Flask(__name__)


def _register_source_routes():
    """Copy every API route from the source app, stripping the /api prefix."""
    for rule in list(source_app.url_map.iter_rules()):
        path = rule.rule
        # Root page -> served as static index.html by EdgeOne.
        # Static assets -> served directly by EdgeOne from /static.
        if path == "/" or path.startswith("/static"):
            continue
        # Strip the leading /api that EdgeOne removes before dispatch.
        new_path = re.sub(r"^/api", "", path) or "/"
        view_func = source_app.view_functions[rule.endpoint]
        methods = [m for m in rule.methods if m not in ("HEAD", "OPTIONS")]
        if not methods:
            methods = ["GET"]
        app.add_url_rule(new_path, "eo_" + rule.endpoint, view_func, methods=methods)


_register_source_routes()


@app.route("/")
def api_root():
    """Friendly probe at the API base (e.g. GET /api/ on EdgeOne)."""
    return {
        "status": "ok",
        "service": "stock-analyzer",
        "platform": "edgeone",
        "note": "All JSON API endpoints are served under /api/*",
    }


if __name__ == "__main__":
    # Local debugging only; EdgeOne invokes the WSGI `app` object.
    app.run(host="0.0.0.0", port=8088, debug=True)
