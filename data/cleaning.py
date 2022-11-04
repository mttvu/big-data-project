import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def groupby_frequency(df, frequency, x, y):
    """
    Returns the dataframe grouped by hour.
    
    Parameters
    ----------
    df : DataFrame
        The dataframe to be grouped.

    frequency : str
        The frequency to group by. For example: 'H' for hour.

    x : str
        The date column.

    y : str
        The value column to group.
    
    """
    grouped = df.groupby(pd.Grouper(key=x, freq=frequency))[y].mean()
    grouped = {x: grouped.keys(), y: grouped}
    return pd.DataFrame(grouped)


def remove_outliers(df, target_column, window, threshold):
    """
    Remove outliers from time series data using a rolling z-score.
    
    Parameters
    ----------
    df : DataFrame
        The dataframe to be cleaned.

    target_column : str
        The column to be cleaned.

    window : str
        The window to calculate zscore on. For example: '7D' for a window of seven days

    threshold : int
        z-score threshold
    """
    roll = df[target_column].rolling(window)
    z_scores = (df[target_column] - roll.mean()) / roll.std()

    z_scores = abs(z_scores)
    filtered_entries = (z_scores < threshold)
    clean_df = df.copy()
    clean_df.loc[~filtered_entries, target_column] = None
    return clean_df


if __name__ == "__main__":
    wd_df = pd.read_pickle('lobith_waterdepth.pkl')
    wd_df.columns = ['time','lat','lng','y']
    wd_df = wd_df[['time', 'y']]

    wd_df = groupby_frequency(
        df=wd_df, 
        frequency='H',
        x='time',
        y='y')
    wd_df.set_index('time', inplace=True)

    wd_df_cleaned = remove_outliers(
        df=wd_df, 
        target_column='y', 
        window='7D', 
        threshold=1.5)

    # wd_df_cleaned = wd_df_cleaned.interpolate()
    # wd_df['moving_average'] = wd_df['y'].rolling('1D').mean()

    fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=wd_df.index, 
    #     y=wd_df.y,
    #     name='original'))
    fig.add_trace(go.Scatter(
        x=wd_df_cleaned.index, 
        y=wd_df_cleaned.y,
        name='cleaned'))
    fig.show()