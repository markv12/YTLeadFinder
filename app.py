import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import youtube_utils
import storage_utils
import openai_utils
import ui_components
import sheets_utils
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Music Game Lead Finder", layout="wide")

def main():
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

    tabs = st.tabs(["Search YouTube", "Channel Database", "Good Fit Channels", "Skipped Channels"])

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
                    pub_after_str = f"{published_after}T00:00:00Z"
                    
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
        
        # Process approvals in a dedicated placeholder to avoid greying out everything
        proc_placeholder = st.empty()
        if "processing_id" in st.session_state:
            # Re-fetch searches to get latest state
            all_s = storage_utils.get_all_searches()
            search_to_process = next((s for s in all_s if s.get('timestamp') == st.session_state.processing_id), None)
            if search_to_process:
                with proc_placeholder:
                    with st.spinner(f"Fetching full channel data for '{search_to_process['query']}'..."):
                        # Batch fetch channel stats to save quota (1 unit per 50 channels)
                        channel_ids_to_fetch = list(set(res['channelId'] for res in search_to_process['results']))
                        
                        # Filter out channels we already have data for
                        channel_ids_to_fetch = [cid for cid in channel_ids_to_fetch if not storage_utils.get_channel_data(cid)]
                        
                        if channel_ids_to_fetch:
                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            
                            def process_channel(stats):
                                cid = stats['id']
                                # Get recent videos
                                videos = youtube_utils.get_recent_channel_videos(stats['uploadsPlaylistId'])
                                # Get stats for all videos
                                v_ids = [v['id'] for v in videos]
                                v_stats = youtube_utils.batch_get_video_stats(v_ids)
                                # Merge
                                for v in videos:
                                    v_stat = v_stats.get(v['id'], {})
                                    v['viewCount'] = v_stat.get('viewCount', 0)
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
                                        future.result()
                                    
                        st.success(f"Search '{search_to_process['query']}' approved and channel data fetched!")
            
            del st.session_state.processing_id
            st.rerun()

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
                    with st.expander(f"{s['query']}", key=f"pending_exp_{actual_index}"):
                        st.write(f"Results: {len(s['results'])} videos")
                        
                        if st.button(f"Approve Search", key=f"app_{actual_index}"):
                            storage_utils.approve_search(actual_index)
                            st.session_state.processing_id = s.get('timestamp')
                            st.rerun()
                        
                        if st.button(f"Delete Search", key=f"del_{actual_index}"):
                            storage_utils.delete_search(actual_index)
                            st.success("Search deleted.")
                            st.rerun()
                        
                        # Show results preview
                        df_res = pd.DataFrame(s['results'])
                        st.dataframe(
                            df_res[["title", "channelTitle", "publishedAt"]], 
                            column_config={
                                "publishedAt": st.column_config.DatetimeColumn("Published At", format="YYYY-MM-DD")
                            },
                            width="content", 
                            height=(len(df_res) + 1) * 35 + 3
                        )

            st.header("✅ Approved Searches")
            if not approved:
                st.info("No approved searches.")
            else:
                for actual_index, s in reversed(approved):
                    with st.expander(f"{s['query']}", key=f"approved_exp_{actual_index}"):
                        st.write(f"Results: {len(s['results'])} videos")
                        
                        if st.button(f"Delete Search", key=f"del_app_{actual_index}"):
                            storage_utils.delete_search(actual_index)
                            st.success("Search deleted.")
                            st.rerun()
                        
                        # Show results preview
                        df_res = pd.DataFrame(s['results'])
                        st.dataframe(
                            df_res[["title", "channelTitle", "publishedAt"]], 
                            column_config={
                                "publishedAt": st.column_config.DatetimeColumn("Published At", format="YYYY-MM-DD")
                            },
                            width="content", 
                            height=(len(df_res) + 1) * 35 + 3
                        )

    # --- Tab 2: Master Channel Database ---
    with tabs[1]:
        st.header("📊 Channel Database")

        # Only show channels with no status (None)
        channel_ids = storage_utils.get_channels_by_status(None)

        if not channel_ids:
            st.info("No new channels to review. Run a search or check other tabs.")
        else:
            st.write(f"Showing **{len(channel_ids)}** channels pending review.")

            # Ranking Logic
            rank_query = st.text_input("Enter a query to rank channels by relevance (e.g., 'DAW workflow for beginners')", key="db_rank_query")

            if "last_rankings" not in st.session_state:
                st.session_state.last_rankings = []

            if st.button("Rank Channels", key="db_rank_btn"):
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
                            valid_cids = [cid for cid in channel_ids if cid in embeddings]
                            if not valid_cids:
                                st.warning("No embeddings found for ranking.")
                            else:
                                # Convert to matrix for fast computation
                                query_matrix = np.array([query_emb])
                                embs_matrix = np.array([embeddings[cid] for cid in valid_cids])

                                # Single matrix operation
                                sim_scores = cosine_similarity(query_matrix, embs_matrix)[0]

                                for cid, score in zip(valid_cids, sim_scores):
                                    c_data = storage_utils.get_channel_data(cid)
                                    new_rankings.append({
                                        "ID": cid,
                                        "Channel": c_data["title"],
                                        "Similarity Score": round(float(score), 4),
                                        "Subs": c_data.get("subscriberCount"),
                                        "Videos": c_data.get("videoCount"),
                                        "URL": f"https://youtube.com/{c_data.get('customUrl', '')}"
                                    })
                                st.session_state.last_rankings = new_rankings

            # Determine which data to show
            if st.session_state.last_rankings:
                # Filter rankings to only show channels currently in the filtered master list
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

            # Side-by-Side Layout
            col_main, col_sim = st.columns(2)

            db_table_key = "channel_db_table"
            with col_main:
                event = ui_components.render_channel_table(df, show_selection=True, key=db_table_key)

            if event and event.selection.rows:
                selected_idx = event.selection.rows[0]
                if selected_idx < len(df):
                    target_row = df.iloc[selected_idx]
                    target_id = target_row["ID"]
                    target_name = target_row["Channel"]

                    with col_sim:
                        st.subheader(f"🛠️ Actions for {target_name}")
                        action_c1, action_c2 = st.columns(2)

                        with action_c1:
                            if st.button("✅ Mark as Good Fit", use_container_width=True):
                                storage_utils.set_channel_status(target_id, "good_fit")
                                st.toast(f"✅ {target_name} marked as Good Fit!")
                                st.rerun()
                        with action_c2:
                            if st.button("❌ Mark as Skip", use_container_width=True):
                                storage_utils.set_channel_status(target_id, "skip")
                                st.toast(f"❌ {target_name} marked as Skip!")
                                st.rerun()

                        st.divider()
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
                else:
                    st.rerun() # Selection is stale
            else:
                with col_sim:
                    st.info("💡 Click anywhere on a row to select a channel and see actions/similarities.")

    # --- Tab 3: Good Fit Channels ---
    with tabs[2]:
        st.header("✅ Good Fit Channels")

        good_fit_ids = storage_utils.get_channels_by_status("good_fit")

        if not good_fit_ids:
            st.info("No channels marked as 'Good Fit' yet.")
        else:
            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.write(f"You have identified **{len(good_fit_ids)}** high-quality leads.")
            with col_h2:
                if st.button("📤 Sync to Google Sheets", use_container_width=True):
                    # Prepare list of channel data for syncing
                    channels_to_sync = []
                    for cid in good_fit_ids:
                        c_data = storage_utils.get_channel_data(cid)
                        if c_data:
                            channels_to_sync.append({
                                "ID": cid,
                                "Channel": c_data["title"]
                            })
                    
                    if channels_to_sync:
                        with st.spinner("Syncing to Google Sheets..."):
                            success, message = sheets_utils.sync_good_fit_channels(channels_to_sync)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    else:
                        st.warning("No channel data found to sync.")

            rankings = []
            for cid in good_fit_ids:
                c_data = storage_utils.get_channel_data(cid)
                if c_data:
                    rankings.append({
                        "ID": cid,
                        "Channel": c_data["title"],
                        "Subs": c_data.get("subscriberCount"),
                        "Videos": c_data.get("videoCount"),
                        "URL": f"https://youtube.com/{c_data.get('customUrl', '')}"
                    })
            df_good = pd.DataFrame(rankings)

            col_good_main, col_good_action = st.columns([2, 1])
            good_table_key = "good_fit_table"

            with col_good_main:
                event = ui_components.render_channel_table(df_good, show_selection=True, key=good_table_key)

            if event and event.selection.rows:
                selected_idx = event.selection.rows[0]
                if selected_idx < len(df_good):
                    target_row = df_good.iloc[selected_idx]
                    target_id = target_row["ID"]
                    target_name = target_row["Channel"]

                    with col_good_action:
                        st.subheader(f"🛠️ Actions")
                        if st.button(f"🔄 Move '{target_name}' back to Database", use_container_width=True):
                            storage_utils.set_channel_status(target_id, None)
                            st.toast(f"🔄 {target_name} moved back to database.")
                            st.rerun()
                else:
                    st.rerun()

    # --- Tab 4: Skipped Channels ---
    with tabs[3]:
        st.header("❌ Skipped Channels")

        skipped_ids = storage_utils.get_channels_by_status("skip")

        if not skipped_ids:
            st.info("No channels skipped yet.")
        else:
            st.write(f"Showing **{len(skipped_ids)}** skipped channels.")

            rankings = []
            for cid in skipped_ids:
                c_data = storage_utils.get_channel_data(cid)
                if c_data:
                    rankings.append({
                        "ID": cid,
                        "Channel": c_data["title"],
                        "Subs": c_data.get("subscriberCount"),
                        "Videos": c_data.get("videoCount"),
                        "URL": f"https://youtube.com/{c_data.get('customUrl', '')}"
                    })
            df_skip = pd.DataFrame(rankings)

            col_skip_main, col_skip_action = st.columns([2, 1])
            skip_table_key = "skip_table"

            with col_skip_main:
                event = ui_components.render_channel_table(df_skip, show_selection=True, key=skip_table_key)

            if event and event.selection.rows:
                selected_idx = event.selection.rows[0]
                if selected_idx < len(df_skip):
                    target_row = df_skip.iloc[selected_idx]
                    target_id = target_row["ID"]
                    target_name = target_row["Channel"]

                    with col_skip_action:
                        st.subheader(f"🛠️ Actions")
                        if st.button(f"🔄 Move '{target_name}' back to Database", use_container_width=True):
                            storage_utils.set_channel_status(target_id, None)
                            st.toast(f"🔄 {target_name} moved back to database.")
                            st.rerun()
                else:
                    st.rerun()

if __name__ == "__main__":
    main()
