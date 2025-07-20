from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/api/users/{user_id}')
def get_user(user_id: int):
    return {"user_id": user_id}