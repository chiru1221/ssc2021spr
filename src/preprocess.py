import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def convert_notation(df):
    split_tempo = lambda x: x.split('-')
    df['tempo_min'] = df.tempo.apply(split_tempo).apply(lambda x: x[0]).astype(np.int64)
    df['tempo_max'] = df.tempo.apply(split_tempo).apply(lambda x: x[1]).astype(np.int64)
    df.tempo = df.tempo_max - df.tempo_min
    return df

def region_encoding(train_df, test_df):
    le = LabelEncoder()
    regions = test_df.region.unique()
    train_df = train_df[train_df.region.isin(regions)]
    # train_df = train_df[train_df.region != 'region_M']
    train_df = train_df.reset_index(drop=True)
    test_df.region = le.fit_transform(test_df.region)
    train_df.region = le.transform(train_df.region)
    # print(le.transform(['unknown']))
    return train_df, test_df, le

def region_fix(x, y, thre_count, encode):
    df = pd.concat([x, y], axis=1)
    group = df.groupby(['region', 'genre']).count().iloc[:, 0].reset_index()
    for _, row in group[group.iloc[:, -1] <= thre_count].iloc[:, :2].iterrows():
        index = df.query('region == @row.region & genre == @row.genre').index.values
        x.iloc[index].region = encode
    return x

def region_grouping(train_df, test_df, target_cols, diff=True):
    df = pd.concat([train_df, test_df])
    target_cols.append('region')
    group = df[target_cols].groupby('region').agg([np.mean, np.std]).reset_index()
    group.columns = ['_'.join(x) for x in group.columns.ravel()]
    if diff:
        target_cols.remove('region')
        def diff_mean_diff_std(df):
            df = df.merge(group, how='left', left_on='region', right_on='region_')
            df = df.drop('region_', axis=1)
            for col in target_cols:
                df['{0}_region_mean_diff'.format(col)] = df[col] - df['{0}_mean'.format(col)]
                df['{0}_region_mean_diff_region_std_diff'.format(col)] = df['{0}_region_mean_diff'.format(col)] - df['{0}_std'.format(col)]
            return df
        return diff_mean_diff_std(train_df), diff_mean_diff_std(test_df)
    else:
        return group

