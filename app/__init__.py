"""Resume Optimizer Flask application factory."""

import secrets

from flask import Flask

from app.config import Config


def create_app(config_class: type = Config) -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(config_class)

    # Multi-worker deployments must configure one shared FLASK_SECRET_KEY.
    if app.config.get("SECRET_KEY") in {None, "", "dev-only-change-me", "change-me-in-production"}:
        app.config["SECRET_KEY"] = secrets.token_hex(32)
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
    if app.config.get("SESSION_COOKIE_SAMESITE") is None:
        app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    if app.config.get("MAX_CONTENT_LENGTH") is None:
        app.config["MAX_CONTENT_LENGTH"] = 600_000
    app.config.setdefault("ARTIFACT_TTL_SECONDS", 3600)
    app.config.setdefault("ARTIFACT_MAX_ENTRIES", 64)
    app.config.setdefault("ARTIFACT_MAX_BYTES", 32_000_000)

    from app.artifacts import ArtifactStore

    app.extensions["resume_artifacts"] = ArtifactStore(
        ttl=app.config["ARTIFACT_TTL_SECONDS"],
        max_entries=app.config["ARTIFACT_MAX_ENTRIES"],
        max_bytes=app.config["ARTIFACT_MAX_BYTES"],
    )

    from app.knowledge import KnowledgeStore

    # Tests get an isolated database even when the developer has a populated store.
    db_path = app.config.get("KNOWLEDGE_DB_PATH", Config.KNOWLEDGE_DB_PATH)
    if app.testing and db_path == Config.KNOWLEDGE_DB_PATH:
        db_path = ":memory:"
    app.extensions["knowledge_store"] = KnowledgeStore(db_path)

    from app.routes import bp

    app.register_blueprint(bp)
    return app
