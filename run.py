from pathlib import Path

import uvicorn

from server.tls import ensure_self_signed_cert

BASE_DIR = Path(__file__).resolve().parent
CERT_FILE = BASE_DIR / "certs" / "server.crt"
KEY_FILE = BASE_DIR / "certs" / "server.key"

if __name__ == "__main__":
    ensure_self_signed_cert(CERT_FILE, KEY_FILE)
    uvicorn.run(
        "server.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(BASE_DIR),
        ssl_certfile=str(CERT_FILE),
        ssl_keyfile=str(KEY_FILE),
    )
