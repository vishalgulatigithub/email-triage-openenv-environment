from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, OpenEnv!"}

def main():
    """
    Entrypoint for CLI/script.
    """
    uvicorn.run("server.app:app", host="127.0.0.1", port=8000, reload=True)

# Make sure main is callable when run directly
if __name__ == "__main__":
    main()