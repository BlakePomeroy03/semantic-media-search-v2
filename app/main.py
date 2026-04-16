import os
from fastapi import FastAPI, UploadFile, File, Depends
from app.db import engine, Base, SessionLocal
import app.models as models
from sqlalchemy.orm import Session


app = FastAPI()

Base.metadata.create_all(bind=engine)

os.makedirs("uploads", exist_ok = True)

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
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = f"uploads/{file.filename}"

    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    size = os.path.getsize(file_location)

    new_record = models.FileRecord(
        filename = file.filename,
        filepath = file_location,
        size = size
    )

    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return {"message": "Success", "id": new_record.id}


@app.get("/files")
async def list_files(db: Session = Depends(get_db)):
    files = db.query(models.FileRecord).all()
    return files
