from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = (ROOT_DIR / "trustvision.db").resolve()
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

# Setting check_same_thread to False is needed for FastAPI with SQLite.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
