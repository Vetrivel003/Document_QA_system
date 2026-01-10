import os
from pathlib import Path
from typing import Optional 
from dotenv import load_dotenv

load_dotenv()

class Config:
    #Base Path
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = DATA_DIR / "uploads"
    PROCESSED_DIR = DATA_DIR / "processed"

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY","")

    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    CHROMA_PERSIST_DIR: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    )

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    SUPPORTED_FORMATS = {'.pdf', '.txt', '.docx'}
    MAX_FILE_SIZE_MB = 50

    @classmethod
    def validate(cls) -> None:
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
    
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        print("validated successfully!")
        print(f"Upload directory: {cls.UPLOAD_DIR}")
        print(f"Using model: {cls.GROQ_MODEL}")
        print(f"Using embeddings: {cls.EMBEDDING_MODEL}")

if __name__ == "__main__":
    Config.validate()