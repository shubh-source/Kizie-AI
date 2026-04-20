from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text
from models import SessionLocal, Memory, User, init_db
import warnings
warnings.filterwarnings("ignore") # Ignore some PyTorch warnings for clean output

# Load a fast embedding model (runs locally)
# Using all-MiniLM-L6-v2 for speed and low memory usage (384 dimensions)
print("Loading Embedding Model (SentenceTransformer)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_user(name: str):
    db = SessionLocal()
    user = db.query(User).filter(User.name == name).first()
    if not user:
        user = User(name=name)
        db.add(user)
        db.commit()
        db.refresh(user)
    db.close()
    return user

def save_memory(user_id: int, text_content: str):
    print(f"[*] Saving memory: '{text_content}'")
    embedding = model.encode(text_content).tolist()
    
    db = SessionLocal()
    new_memory = Memory(
        user_id=user_id,
        content=text_content,
        embedding=embedding
    )
    db.add(new_memory)
    db.commit()
    db.close()
    print("    -> Memory saved successfully.")

def search_relevant_memory(user_id: int, query: str, limit: int = 3):
    print(f"[*] Searching memories related to: '{query}'")
    query_embedding = model.encode(query).tolist()
    
    db = SessionLocal()
    # Using L2 distance (<->) for vector similarity search in pgvector
    results = db.query(Memory).filter(
        Memory.user_id == user_id
    ).order_by(
        Memory.embedding.l2_distance(query_embedding)
    ).limit(limit).all()
    
    db.close()
    return [res.content for res in results]

if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
    
    user = create_user("User")
    print(f"User loaded: {user.name} (ID: {user.id})\n")
    
    # Save a test memory
    save_memory(user.id, "Kal main office mein bohot pareshan tha boss ki wajah se.")
    save_memory(user.id, "Mera favourite color neon pink aur teal hai, cyberpunk vibes.")
    save_memory(user.id, "Mujhe dogs bilkul pasand nahi hain, mujhe cats achhi lagti hain.")
    
    # Search memory
    print("\n--- Testing Memory Recall ---")
    
    # Test 1
    query_1 = "Mujhe kaunse rang pasand hain?"
    results_1 = search_relevant_memory(user.id, query_1, limit=1)
    print(f"Result for '{query_1}': {results_1[0]}")
    
    # Test 2
    query_2 = "Office mein kal kya hua tha?"
    results_2 = search_relevant_memory(user.id, query_2, limit=1)
    print(f"Result for '{query_2}': {results_2[0]}")
