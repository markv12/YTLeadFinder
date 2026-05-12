import streamlit as st
import pandas as pd

def render_channel_table(df: pd.DataFrame):
    """
    Renders a standard channel data table with specific column configurations.
    
    Args:
        df (pd.DataFrame): Dataframe containing channel information. 
                          Expected columns: 'Channel', 'Subs', 'Videos', 'URL'.
                          Optional column: 'Similarity Score'.
    """
    if df.empty:
        st.info("No channel data to display.")
        return

    # Wrap in columns to prevent the table from stretching across the wide layout
    col_tbl, _ = st.columns([3, 1])
    with col_tbl:
        # Dynamic column config based on whether similarity score is present
        col_cfg = {
            "Channel": st.column_config.TextColumn(width="medium"),
            "Subs": st.column_config.NumberColumn(width="medium"),
            "Videos": st.column_config.NumberColumn(width="medium"),
            "URL": st.column_config.LinkColumn("YouTube Link", display_text="Visit Channel", width="medium")
        }
        if "Similarity Score" in df.columns:
            col_cfg["Similarity Score"] = st.column_config.NumberColumn(format="%.4f", width="medium")

        st.dataframe(
            df, 
            hide_index=True,
            use_container_width=False,
            column_config=col_cfg,
            height=(len(df) + 1) * 35 + 3
        )
