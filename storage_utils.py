import json
import os
import numpy as np
from datetime import datetime

DATA_DIR = "data"
SEARCHES_FILE = os.path.join(DATA_DIR, "searches.jsonl")
CHANNELS_DIR = os.path.join(DATA_DIR, "channels")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npz")

def ensure_dirs():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(CHANNELS_DIR):
        os.makedirs(CHANNELS_DIR)

def save_search(query, results):
    ensure_dirs()
    search_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "results": results,  # List of video metadata + channel IDs
        "approved": False
    }
    with open(SEARCHES_FILE, "a") as f:
        f.write(json.dumps(search_entry) + "\n")
    return search_entry

def get_all_searches():
    if not os.path.exists(SEARCHES_FILE):
        return []
    searches = []
    with open(SEARCHES_FILE, "r") as f:
        for line in f:
            if line.strip():
                searches.append(json.loads(line))
    return searches

def delete_search(index):
    searches = get_all_searches()
    if 0 <= index < len(searches):
        del searches[index]
        with open(SEARCHES_FILE, "w") as f:
            for s in searches:
                f.write(json.dumps(s) + "\n")
        return True
    return False

def approve_search(index):
    searches = get_all_searches()
    if 0 <= index < len(searches):
        searches[index]["approved"] = True
        with open(SEARCHES_FILE, "w") as f:
            for s in searches:
                f.write(json.dumps(s) + "\n")
        return searches[index]
    return None

def save_channel_data(channel_id, data):
    ensure_dirs()
    file_path = os.path.join(CHANNELS_DIR, f"{channel_id}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_channel_data(channel_id):
    file_path = os.path.join(CHANNELS_DIR, f"{channel_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

def get_master_channel_ids():
    searches = get_all_searches()
    channel_ids = set()
    for s in searches:
        if s.get("approved"):
            for res in s.get("results", []):
                channel_ids.add(res["channelId"])
    return list(channel_ids)

def save_embeddings(embeddings_dict):
    ensure_dirs()
    if not embeddings_dict:
        return
    
    ids = list(embeddings_dict.keys())
    # Convert to float32 to save ~50% space vs default float64 with minimal precision loss
    embs = np.array([embeddings_dict[cid] for cid in ids], dtype=np.float32)
    np.savez_compressed(EMBEDDINGS_FILE, ids=ids, embs=embs)

def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with np.load(EMBEDDINGS_FILE, allow_pickle=True) as data:
                ids = data['ids']
                embs = data['embs']
                return {cid: emb for cid, emb in zip(ids, embs)}
        except Exception:
            return {}
    return {}
