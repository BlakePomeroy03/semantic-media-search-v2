import os
import io
import json
from fastapi import FastAPI, UploadFile, File, Depends, Request
from PIL import Image
from app.db import engine, Base, SessionLocal
import app.models as models
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer, util


app = FastAPI()

Base.metadata.create_all(bind=engine)

os.makedirs("uploads", exist_ok = True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Model")

    app.state.model = SentenceTransformer('clip-ViT-B-32')

    print("Done")

    yield

app = FastAPI(lifespan = lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
    

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


@app.get("/search")
async def search_files(query: str, request: Request, db: Session = Depends(get_db)):
    model = request.app.state.model
    query_embedding = model.encode([query])[0]

    all_files = db.query(models.FileRecord).all()

    results = []

    for file_record in all_files:
        image_embedding = json.loads(file_record.embedding)
        score = util.cos_sim(image_embedding, query_embedding)[0][0].item()
        results.append({"filename": file_record.filename, "score": score})


    results.sort(key=lambda x: x["score"], reverse = True)
    return results[:3]
