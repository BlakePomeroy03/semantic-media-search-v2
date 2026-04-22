import os
import io
import json
from fastapi import FastAPI, UploadFile, File, Depends, Request
from PIL import Image
from app.db import engine, Base, SessionLocal,get_db
import app.models as models
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer, util
from app.routes import search
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

Base.metadata.create_all(bind=engine)

os.makedirs("uploads", exist_ok = True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Model")
    app.state.model = SentenceTransformer('clip-ViT-L-14')
    print("Preparing Cache")
    db = SessionLocal()

    try:
        all_files = db.query(models.FileRecord).all()

        cache_filenames = []
        cache_vectors = []
        count = 0
        for file_record in all_files:
            cache_filenames.append(file_record.filename)

            vector = json.loads(file_record.embedding)
            cache_vectors.append(vector)

            app.state.filenames = cache_filenames
            app.state.vectors = cache_vectors

            count = count + 1

    finally:
        print(f"Cache warmed with {count} vectors.")
        db.close()

    yield

app = FastAPI(lifespan = lifespan)

app.mount("/images", StaticFiles(directory="uploads"), name = "images")

app.include_router(search.router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile, db: Session = Depends(get_db)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert('RGB')

    model = request.app.state.model

    embedding = model.encode([image])[0].tolist()

    embedding_json = json.dumps(embedding)

    print(f"File {file.filename} vector: {embedding[:5]}...")

    file_location = f"uploads/{file.filename}"

    with open(file_location, "wb") as buffer:
        buffer.write(contents)

    size = os.path.getsize(file_location)

    new_record = models.FileRecord(
        filename = file.filename,
        filepath = file_location,
        embedding = embedding_json,
        size = size
    )

    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return {"filename": file.filename, "message": "Vector extracted successfully!"}


@app.get("/files")
async def list_files(db: Session = Depends(get_db)):
    files = db.query(models.FileRecord).all()
    return files

@app.get("/")
async def frontend():
    return FileResponse("frontend/index.html")