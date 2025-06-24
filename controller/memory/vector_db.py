from sentence_transformers import SentenceTransformer
import chromadb


model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.Client()
collection = client.create_collection(name="scene_memory")

# Add a new scene
tasks = ["Find the nearest object that looks like a laptop and tell me its label.",
"Approach either a cup or a mug and scan it.",
"Look for something red and report what it is.",
"If you see a book or a notebook, go to it and take a picture.",
"Search for a bottle, and once found, say what’s written on it.",
"Identify any visible fruit and tell me how big it is.",
"Scan the area and list any chairs you find.",
"Move to either a can or a glass and describe its height.",
"Look around for a dog or a cat and scan it.",
"If you spot any electronic device, take a picture and report its label."]


for num, task in enumerate(tasks):
    embedding = model.encode(task)
    collection.add(
        documents=[task],
        embeddings=[embedding.tolist()],
        ids=[f"task{num}"]
    )

user_task = input("Write a task: ")
user_task_embedding = model.encode(user_task)

# Search for similar scenes
results = collection.query(query_embeddings=[user_task_embedding.tolist()], n_results=3)

for i in range(len(results["ids"][0])):
    print(f"Result {i}: {results['documents'][0][i]} (Distance: {results['distances'][0][i]})")