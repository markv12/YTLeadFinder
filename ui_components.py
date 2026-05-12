import streamlit as st
import pandas as pd

def render_channel_table(df: pd.DataFrame, show_selection: bool = True):
    """
    Renders a standard channel data table with specific column configurations.
    
    Args:
        df (pd.DataFrame): Dataframe containing channel information. 
                          Expected columns: 'Channel', 'Subs', 'Videos', 'URL'.
                          Optional columns: 'Similarity Score', 'ID'.
        show_selection (bool): Whether to enable row selection.
    
    Returns:
        The selection event from st.dataframe (or None if selection is disabled).
    """
    if df.empty:
        st.info("No channel data to display.")
        return None

    # Dynamic column config
    col_cfg = {
        "Channel": st.column_config.TextColumn(width=200),
        "Subs": st.column_config.NumberColumn(width=80),
        "Videos": st.column_config.NumberColumn(width=80),
        "URL": st.column_config.LinkColumn("YouTube Link", display_text="Visit Channel", width=120),
        "ID": None # Hide the ID column
    }
    
    if "Similarity Score" in df.columns:
        col_cfg["Similarity Score"] = st.column_config.NumberColumn(format="%.4f", width=120)

    # Configure selection mode
    selection_mode = "single-row" if show_selection else []
    on_select = "rerun" if show_selection else "ignore"

    # Use 'width' instead of 'use_container_width' to avoid deprecation warnings
    # 'content' mimics use_container_width=False
    event = st.dataframe(
        df, 
        hide_index=True,
        width="content",
        column_config=col_cfg,
        height=(len(df) + 1) * 35 + 3,
        selection_mode=selection_mode,
        on_select=on_select
    )
    
    return event if show_selection else None
