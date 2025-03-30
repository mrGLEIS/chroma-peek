import chromadb
import pandas as pd

class ChromaPeek:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path)

    ## Function that returns all collection names
    def get_collections(self):
        return [col for col in self.client.list_collections()]  # ChromaDB v0.6.0 returns a list of strings

    ## Function to return documents/data inside the collection
    def get_collection_data(self, collection_name, dataframe=False):
        collection = self.client.get_collection(collection_name)
        data = collection.get(include=["documents", "metadatas", "embeddings"])  # Ensure all data fields are retrieved
        
        # Flatten the dictionary output into a structured format
        records = []
        for i in range(len(data["ids"])):
            records.append({
                "id": data["ids"][i],
                "document": data["documents"][i] if "documents" in data else None,
                "metadata": data["metadatas"][i] if "metadatas" in data else None,
                "embedding": data["embeddings"][i] if "embeddings" in data else None
            })
        
        if dataframe:
            return pd.DataFrame(records)
        return records
    
    ## Function to query the selected collection
    def query(self, query_str, collection_name, k=3, dataframe=False):
        collection = self.client.get_collection(collection_name)
        res = collection.query(
            query_texts=[query_str], 
            n_results=min(k, len(collection.get()["ids"]))  # Ensure n_results is within valid range
        )

        # Convert output to a structured format
        results = []
        for i in range(len(res["ids"][0])):  # res["ids"] is a nested list
            results.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i] if "documents" in res else None,
                "metadata": res["metadatas"][0][i] if "metadatas" in res else None,
                "embedding": res["embeddings"][0][i] if "embeddings" in res else None,
                "distance": res["distances"][0][i] if "distances" in res else None
            })

        if dataframe:
            return pd.DataFrame(results)
        return results
