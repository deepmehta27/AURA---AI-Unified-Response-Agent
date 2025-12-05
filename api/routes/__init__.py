"""
API Routes Package This makes the routes folder a package and exposes all routers for easy import.
"""

from api.routes import health, query, upload

__all__ = ["health", "query", "upload"]
