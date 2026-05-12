import os
import logging
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

_tokenizer = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 8191

def _truncate_to_tokens(text: str) -> str:
    tokens = _tokenizer.encode(text)
    if len(tokens) <= _MAX_TOKENS:
        return text
    return _tokenizer.decode(tokens[:_MAX_TOKENS])

def get_embeddings_batch(texts, model="text-embedding-3-large"):
    """
    Fetch embeddings for a list of strings in a single request.
    Max 2048 inputs per request (OpenAI limit).
    """
    if not client or not texts: return []

    logger.info(f"OpenAI Request: embeddings.create(model='{model}', input_count={len(texts)})")
    truncated_texts = [_truncate_to_tokens(t) for t in texts]

    response = client.embeddings.create(input=truncated_texts, model=model)
    return [item.embedding for item in response.data]

def get_embedding(text, model="text-embedding-3-large"):
    """Fetch a single embedding."""
    results = get_embeddings_batch([text], model=model)
    return results[0] if results else None

def get_selected_videos(channel_data, limit=50):
    """Selects the top/recent videos from channel data using the defined logic."""
    all_videos = channel_data.get("videos", [])
    if not all_videos:
        return []

    # Defensive check: ensure all videos have at least a title
    valid_videos = []
    for v in all_videos:
        if not isinstance(v, dict):
            continue
        
        title = v.get('title')
        if title:
            # Fallback for ID: use videoId, id, or the title itself if both are missing
            vid = v.get('id') or v.get('videoId') or title
            if 'id' not in v:
                v['id'] = vid
            valid_videos.append(v)

    if not valid_videos:
        return []

    # Sort by recent
    recent_videos = sorted(valid_videos, key=lambda x: x.get('publishedAt', ''), reverse=True)[:limit]
    
    # Sort by view count
    most_viewed = sorted(valid_videos, key=lambda x: x.get('viewCount', 0), reverse=True)[:limit]
    
    # Combine and deduplicate by ID
    selected_videos = []
    seen_ids = set()
    
    for v in (recent_videos + most_viewed):
        vid = v.get('id')
        if vid and vid not in seen_ids:
            selected_videos.append(v)
            seen_ids.add(vid)
            if len(selected_videos) >= limit:
                break
                
    return selected_videos

def create_channel_profile_text(channel_data):
    """Combines only video titles into a single string for embedding."""
    selected_videos = get_selected_videos(channel_data)
    if not selected_videos:
        return ""
    
    # Just concatenate titles with a separator
    titles = [v['title'] for v in selected_videos if 'title' in v]
    return " | ".join(titles)
