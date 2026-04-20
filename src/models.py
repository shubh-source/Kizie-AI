from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    memories = relationship("Memory", back_populates="user")

class Memory(Base):
    __tablename__ = 'memories'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384)) # Using 384 dimensions for all-MiniLM-L6-v2 model
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="memories")

# Database connection
DB_URL = os.getenv("DATABASE_URL", "postgresql://kizie_user:kizie_password@localhost:5432/kizie_db")
engine = create_engine(DB_URL)

def init_db():
    # Make sure pgvector extension exists
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)
