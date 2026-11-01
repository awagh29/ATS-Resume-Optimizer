"""Local entry point, served by Waitress unless explicit debug is enabled."""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    if app.config["DEBUG"]:
        app.run(host=host, port=app.config["PORT"], debug=True)
    else:
        from waitress import serve

        serve(app, host=host, port=app.config["PORT"], threads=4,
              channel_timeout=600, max_request_body_size=app.config["MAX_CONTENT_LENGTH"])
