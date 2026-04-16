from sqlalchemy import Column, Integer, String
from app.db import Base

class FileRecord(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    filepath = Column(String, unique=True)
    size = Column(Integer)
