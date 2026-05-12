import os
import logging
import threading
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Use thread-local storage for the YouTube client to ensure thread safety
_thread_local = threading.local()

def get_youtube_client():
    if not API_KEY:
        return None
    if not hasattr(_thread_local, "youtube"):
        # The build() function creates an httplib2.Http object which is NOT thread-safe.
        # By building it here, each thread gets its own instance.
        _thread_local.youtube = build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)
    return _thread_local.youtube

def search_videos(query, max_results=25, order="relevance", published_after=None, video_duration="any"):
    youtube = get_youtube_client()
    if not youtube: return []
    
    params = {
        "q": query,
        "part": "snippet",
        "type": "video",
        "maxResults": max_results,
        "order": order
    }
    
    if published_after:
        params["publishedAfter"] = published_after
        
    if video_duration != "any":
        params["videoDuration"] = video_duration
        
    logger.info(f"YouTube Request: search.list({params})")
    request = youtube.search().list(**params)
    response = request.execute()
    results = []
    for item in response.get("items", []):
        results.append({
            "videoId": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channelId": item["snippet"]["channelId"],
            "channelTitle": item["snippet"]["channelTitle"],
            "description": item["snippet"]["description"],
            "publishedAt": item["snippet"]["publishedAt"]
        })
    return results

def get_channel_stats(channel_id):
    """Fetch stats for a single channel."""
    stats = batch_get_channel_stats([channel_id])
    return stats[0] if stats else {}

def batch_get_channel_stats(channel_ids):
    """
    Fetch stats for multiple channels in a single request (max 50).
    Cost: 1 quota unit.
    """
    youtube = get_youtube_client()
    if not youtube or not channel_ids: return []
    
    logger.info(f"YouTube Request: channels.list(ids_count={len(channel_ids)})")
    # YouTube allows max 50 IDs per request
    ids_str = ",".join(channel_ids[:50])
    request = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        id=ids_str
    )
    response = request.execute()
    
    results = []
    for item in response.get("items", []):
        results.append({
            "id": item["id"],
            "title": item["snippet"]["title"],
            "description": item["snippet"]["description"],
            "customUrl": item["snippet"].get("customUrl"),
            "subscriberCount": item["statistics"].get("subscriberCount"),
            "viewCount": item["statistics"].get("viewCount"),
            "videoCount": item["statistics"].get("videoCount"),
            "uploadsPlaylistId": item["contentDetails"]["relatedPlaylists"]["uploads"]
        })
    return results

from concurrent.futures import ThreadPoolExecutor, as_completed

def get_all_channel_videos(uploads_playlist_id):
    youtube = get_youtube_client()
    if not youtube: return []
    videos = []
    
    # First, get the first page to see total results/tokens
    logger.info(f"YouTube Request: Fetching video list for playlist {uploads_playlist_id}")
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=50
    )
    response = request.execute()
    
    for item in response.get("items", []):
        videos.append({
            "id": item["contentDetails"]["videoId"],
            "title": item["snippet"]["title"],
            "publishedAt": item["snippet"]["publishedAt"][:10]
        })
        
    next_page_token = response.get("nextPageToken")
    if not next_page_token:
        return videos

    while next_page_token:
        request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response.get("items", []):
            videos.append({
                "id": item["contentDetails"]["videoId"],
                "title": item["snippet"]["title"],
                "publishedAt": item["snippet"]["publishedAt"][:10]
            })
        next_page_token = response.get("nextPageToken")
            
    return videos

def _fetch_video_stats_chunk(chunk):
    """Helper for parallel stats fetching."""
    youtube = get_youtube_client()
    request = youtube.videos().list(
        part="statistics",
        id=",".join(chunk)
    )
    response = request.execute()
    chunk_stats = {}
    for item in response.get("items", []):
        chunk_stats[item["id"]] = {
            "viewCount": int(item["statistics"].get("viewCount", 0)),
            "likeCount": int(item["statistics"].get("likeCount", 0)),
            "commentCount": int(item["statistics"].get("commentCount", 0))
        }
    return chunk_stats

def batch_get_video_stats(video_ids):
    """Fetch statistics for videos in parallel batches of 50."""
    youtube = get_youtube_client()
    if not youtube or not video_ids: return {}
    
    stats_map = {}
    chunks = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]
    
    logger.info(f"YouTube Request: videos.list(total_ids={len(video_ids)}) [PARALLEL]")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_chunk = {executor.submit(_fetch_video_stats_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(future_to_chunk):
            chunk_result = future.result()
            stats_map.update(chunk_result)
            
    return stats_map
