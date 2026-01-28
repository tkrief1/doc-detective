import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Use env var if set; default to local docker postgres
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://docd:docd@localhost:5432/docd"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class Base(DeclarativeBase):
    pass
