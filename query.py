from pathway.xpacks.llm.vector_store import VectorStoreClient

# Initialize the client
client = VectorStoreClient(host="localhost", port=8000)

# Define your query text
query_text = "Sample query text"
# Perform the query, retrieving the top 5 nearest neighbors
results = client.query(query=query_text, k=5)

# Print the results
for idx, result in enumerate(results, start=1):
    # print(f"Result {idx}: {result}")
    with open("test.txt", "a") as f:
        f.write(f"Result {idx}: {result}\n")