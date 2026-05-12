import streamlit as st
import pandas as pd

def render_channel_table(df: pd.DataFrame):
    """
    Renders a standard channel data table with specific column configurations.
    
    Args:
        df (pd.DataFrame): Dataframe containing channel information. 
                          Expected columns: 'Channel', 'Subs', 'Videos', 'URL'.
                          Optional columns: 'Similarity Score', 'ID'.
    
    Returns:
        The selection event from st.dataframe.
    """
    if df.empty:
        st.info("No channel data to display.")
        return None

    # Dynamic column config
    col_cfg = {
        "Channel": st.column_config.TextColumn(width="medium"),
        "Subs": st.column_config.NumberColumn(width="medium"),
        "Videos": st.column_config.NumberColumn(width="medium"),
        "URL": st.column_config.LinkColumn("YouTube Link", display_text="Visit Channel", width="medium"),
        "ID": None # Hide the ID column
    }
    
    if "Similarity Score" in df.columns:
        col_cfg["Similarity Score"] = st.column_config.NumberColumn(format="%.4f", width="medium")

    event = st.dataframe(
        df, 
        hide_index=True,
        use_container_width=True,
        column_config=col_cfg,
        height=(len(df) + 1) * 35 + 3,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    return event
