import os
import io
import json
from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
from app.db import engine, Base, SessionLocal, get_db
import app.models as models
from sentence_transformers import SentenceTransformer, util

router = APIRouter()

@router.get("/search")
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
    