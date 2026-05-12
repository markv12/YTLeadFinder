import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import youtube_utils
import storage_utils
import openai_utils
import ui_components
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Music Game Lead Finder", layout="wide")

st.title("🎵 Music Game Lead Finder")

# --- Sidebar: Settings & Maintenance ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("Cache Management")
    
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear Embeddings Cache", help="Force regeneration of OpenAI embeddings"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ This will delete local embeddings. Regenerating them will cost OpenAI credits.")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("✅ Confirm"):
                if os.path.exists(storage_utils.EMBEDDINGS_FILE):
                    os.remove(storage_utils.EMBEDDINGS_FILE)
                    st.success("Cleared!")
                st.session_state.confirm_clear = False
                st.rerun()
        with col_c2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()

if not os.getenv("YOUTUBE_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    st.warning("Please configure your API keys in the `.env` file.")
    st.info("Template provided in `.env.template`")
    st.stop()

tabs = st.tabs(["Search YouTube", "Channel Database"])

# --- Tab 1: Search YouTube ---
with tabs[0]:
    st.header("🔍 Discover New Channels")
    search_query = st.text_input("Enter search query (e.g., 'music production tutorial')")
    
    with st.expander("Advanced Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            search_limit = st.slider("Max Results", 5, 50, 25)
            order_opt = st.selectbox("Sort Order", ["relevance", "date", "viewCount", "rating", "title"], index=0)
        with col2:
            duration_opt = st.selectbox("Video Duration", ["any", "short", "medium", "long"], index=0)
            published_after = st.date_input("Published After", value=None, help="Filter for videos published after this date")

    if st.button("Run Search"):
        with st.spinner("Searching YouTube..."):
            pub_after_str = None
            if published_after:
                pub_after_str = f"{published_after}"
                
            results = youtube_utils.search_videos(
                search_query, 
                max_results=search_limit,
                order=order_opt,
                published_after=pub_after_str,
                video_duration=duration_opt
            )
            if results:
                storage_utils.save_search(search_query, results)
                st.success(f"Found {len(results)} videos. View results below to approve.")
            else:
                st.error("No results found or API error.")

    st.divider()
    searches = storage_utils.get_all_searches()
    
    if not searches:
        st.header("📜 Search History & Approval")
        st.info("No search history.")
    else:
        pending = [(i, s) for i, s in enumerate(searches) if not s['approved']]
        approved = [(i, s) for i, s in enumerate(searches) if s['approved']]

        st.header("⏳ Pending Searches")
        if not pending:
            st.info("No pending searches.")
        else:
            for actual_index, s in reversed(pending):
                with st.expander(f"{s['query']}"):
                    st.write(f"Results: {len(s['results'])} videos")
                    
                    if st.button(f"Approve Search", key=f"app_{actual_index}"):
                        with st.spinner("Fetching full channel data (this may take a while)..."):
                            # Approve the search
                            storage_utils.approve_search(actual_index)
                            
                            # Batch fetch channel stats to save quota (1 unit per 50 channels)
                            channel_ids_to_fetch = list(set(res['channelId'] for res in s['results']))
                            
                            # Filter out channels we already have data for
                            channel_ids_to_fetch = [cid for cid in channel_ids_to_fetch if not storage_utils.get_channel_data(cid)]
                            
                            if channel_ids_to_fetch:
                                from concurrent.futures import ThreadPoolExecutor, as_completed
                                
                                def process_channel(stats):
                                    cid = stats['id']
                                    # Get all videos
                                    videos = youtube_utils.get_all_channel_videos(stats['uploadsPlaylistId'])
                                    # Get stats for all videos (this is already parallelized inside youtube_utils)
                                    v_ids = [v['id'] for v in videos]
                                    v_stats = youtube_utils.batch_get_video_stats(v_ids)
                                    # Merge
                                    for v in videos:
                                        s = v_stats.get(v['id'], {})
                                        v['viewCount'] = s.get('viewCount', 0)
                                    stats['videos'] = videos
                                    storage_utils.save_channel_data(cid, stats)
                                    return cid

                                # Process in chunks of 50 (YouTube limit)
                                for i in range(0, len(channel_ids_to_fetch), 50):
                                    chunk = channel_ids_to_fetch[i:i+50]
                                    batch_stats = youtube_utils.batch_get_channel_stats(chunk)
                                    
                                    # Parallelize the processing of each channel in the chunk
                                    with ThreadPoolExecutor(max_workers=5) as executor:
                                        futures = [executor.submit(process_channel, s) for s in batch_stats]
                                        for future in as_completed(futures):
                                            completed_cid = future.result()
                                        
                            st.success("Search approved and channel data fetched!")
                            st.rerun()
                    
                    if st.button(f"Delete Search", key=f"del_{actual_index}"):
                        storage_utils.delete_search(actual_index)
                        st.success("Search deleted.")
                        st.rerun()
                    
                    # Show results preview
                    df_res = pd.DataFrame(s['results'])
                    st.dataframe(
                        df_res[["title", "channelTitle", "publishedAt"]], 
                        width="content", 
                        height=(len(df_res) + 1) * 35 + 3
                    )

        st.header("✅ Approved Searches")
        if not approved:
            st.info("No approved searches.")
        else:
            for actual_index, s in reversed(approved):
                with st.expander(f"{s['query']}"):
                    st.write(f"Results: {len(s['results'])} videos")
                    
                    if st.button(f"Delete Search", key=f"del_app_{actual_index}"):
                        storage_utils.delete_search(actual_index)
                        st.success("Search deleted.")
                        st.rerun()
                    
                    # Show results preview
                    df_res = pd.DataFrame(s['results'])
                    st.dataframe(
                        df_res[["title", "channelTitle", "publishedAt"]], 
                        width="content", 
                        height=(len(df_res) + 1) * 35 + 3
                    )

# --- Tab 2: Master Channel Database ---
with tabs[1]:
    st.header("📊 Channel Database")
    
    channel_ids = storage_utils.get_master_channel_ids()
    if not channel_ids:
        st.info("No approved channels yet. Run a search and approve results.")
    else:
        st.write(f"Tracking **{len(channel_ids)}** unique channels.")
        
        # Ranking Logic
        rank_query = st.text_input("Enter a query to rank channels by relevance (e.g., 'DAW workflow for beginners')")
        
        if "last_rankings" not in st.session_state:
            st.session_state.last_rankings = []
        
        if st.button("Rank Channels"):
            new_rankings = []
            embeddings = storage_utils.load_embeddings()
            
            # Check for missing channel embeddings
            missing_ids = [cid for cid in channel_ids if cid not in embeddings]
            
            if missing_ids:
                profiles_to_embed = []
                valid_ids = []
                
                for cid in missing_ids:
                    c_data = storage_utils.get_channel_data(cid)
                    if isinstance(c_data, dict) and c_data:
                        profile_text = openai_utils.create_channel_profile_text(c_data)
                        if profile_text:
                            profiles_to_embed.append(profile_text)
                            valid_ids.append(cid)
                
                if profiles_to_embed:
                    with st.spinner(f"Generating embeddings for {len(profiles_to_embed)} channels (batched)..."):
                        for i in range(0, len(profiles_to_embed), 100):
                            text_chunk = profiles_to_embed[i:i+100]
                            id_chunk = valid_ids[i:i+100]
                            
                            batch_embs = openai_utils.get_embeddings_batch(text_chunk)
                            for cid, emb in zip(id_chunk, batch_embs):
                                embeddings[cid] = emb
                                
                        storage_utils.save_embeddings(embeddings)
            
            if rank_query:
                with st.spinner("Ranking..."):
                    query_emb = openai_utils.get_embedding(rank_query)
                    if query_emb:
                        for cid in channel_ids:
                            if cid in embeddings:
                                score = cosine_similarity([query_emb], [embeddings[cid]])[0][0]
                                c_data = storage_utils.get_channel_data(cid)
                                new_rankings.append({
                                    "ID": cid,
                                    "Channel": c_data["title"],
                                    "Similarity Score": round(float(score), 4),
                                    "Subs": c_data.get("subscriberCount"),
                                    "Videos": c_data.get("videoCount"),
                                    "URL": f"https://youtube.com/{c_data.get('customUrl', '')}"
                                })
                        if not new_rankings:
                            st.warning("No embeddings found for ranking.")
                        else:
                            st.session_state.last_rankings = new_rankings

        # Determine which data to show
        if st.session_state.last_rankings:
            # Filter rankings to only show channels currently in the master list
            df = pd.DataFrame([r for r in st.session_state.last_rankings if r['ID'] in channel_ids])
            if not df.empty:
                df = df.sort_values(by="Similarity Score", ascending=False)
        
        # If no rankings or filtered rankings empty, show default view
        if not st.session_state.last_rankings or df.empty:
            rankings = []
            for cid in channel_ids:
                c_data = storage_utils.get_channel_data(cid)
                if c_data:
                    rankings.append({
                        "ID": cid,
                        "Channel": c_data["title"],
                        "Subs": c_data.get("subscriberCount"),
                        "Videos": c_data.get("videoCount"),
                        "URL": f"https://youtube.com/{c_data.get('customUrl', '')}"
                    })
            df = pd.DataFrame(rankings)
        
        # Side-by-Side Layout with spacer to keep them close
        col_main, col_sim, col_spacer = st.columns([1.5, 1.5, 1])
        
        with col_main:
            event = ui_components.render_channel_table(df)
            
        if event and event.selection.rows:
            selected_idx = event.selection.rows[0]
            target_row = df.iloc[selected_idx]
            target_id = target_row["ID"]
            target_name = target_row["Channel"]
            
            with col_sim:
                st.subheader(f"✨ Channels like {target_name}")
                embeddings = storage_utils.load_embeddings()
                similar = openai_utils.find_similar_channels(target_id, embeddings)
                
                if not similar:
                    st.info("No similar channels found in database.")
                else:
                    sim_rankings = []
                    for s in similar:
                        c_data = storage_utils.get_channel_data(s['id'])
                        if c_data:
                            sim_rankings.append({
                                "ID": s['id'],
                                "Channel": c_data["title"],
                                "Similarity Score": s['score'],
                                "Subs": c_data.get("subscriberCount"),
                                "Videos": c_data.get("videoCount"),
                                "URL": f"https://youtube.com/{c_data.get('customUrl', '')}"
                            })
                    
                    if sim_rankings:
                        sim_df = pd.DataFrame(sim_rankings)
                        ui_components.render_channel_table(sim_df, show_selection=False)
