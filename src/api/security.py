import os
from fastapi import Header, HTTPException

def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("API_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="API_KEY is not configured on the server")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
