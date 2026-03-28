import uvicorn
from api import app

def main():
    """Starts the FastAPI server for OpenEnv multi-mode deployment."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
