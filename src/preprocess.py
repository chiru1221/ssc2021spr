import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis

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

def fillna_by_median(train_df, test_df):
    all_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    na_col = all_df.columns[all_df.isna().sum() > 0]
    for col in na_col:
        all_df[col] = all_df[col].fillna(value=all_df[col].median())
    return all_df.iloc[:train_df.shape[0]], all_df.iloc[train_df.shape[0]:]

def region_fix(x, y, thre_count, encode):
    df = pd.concat([x, y], axis=1)
    group = df.groupby(['region', 'genre']).count().iloc[:, 0].reset_index()
    for _, row in group[group.iloc[:, -1] <= thre_count].iloc[:, :2].iterrows():
        index = df.query('region == @row.region & genre == @row.genre').index.values
        x.iloc[index].region = encode
    return x

def region_grouping(train_df, test_df, target_cols, diff=True, drop=False):
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
            
            if drop:
                columns = list(group.columns)
                columns.remove('region_')
                df = df.drop(columns=columns)
            return df
        return diff_mean_diff_std(train_df), diff_mean_diff_std(test_df)
    else:
        return group

def region_additional_group(train_df, test_df, target_cols, drop=False):
    df = pd.concat([train_df, test_df])
    target_cols.append('region')
    group = df[target_cols].groupby('region').agg([kurtosis, skew]).reset_index()
    group.columns = ['_'.join(x) for x in group.columns.ravel()]
    target_cols.remove('region')
    def div_kurtosis_div_skew(df):
        df = df.merge(group, how='left', left_on='region', right_on='region_')
        df = df.drop('region_', axis=1)
        for col in target_cols:
            df['{0}_region_mean_div_kurtosis'.format(col)] = df['{0}_region_mean_diff'.format(col)] / df['{0}_kurtosis'.format(col)]
            df['{0}_region_mean_div_skew'.format(col)] = df['{0}_region_mean_diff'.format(col)] / df['{0}_skew'.format(col)]
        
        if drop:
            columns = list(group.columns)
            columns.remove('region_')
            df = df.drop(columns=columns)
        return df
    return div_kurtosis_div_skew(train_df), div_kurtosis_div_skew(test_df)

def grouping_by_region(train_df, test_df, target_cols):
    df = pd.concat([train_df, test_df])
    target_cols.append('region')
    group = df[target_cols].groupby('region').agg([np.mean, np.std, kurtosis, skew]).reset_index()
    group.columns = ['_'.join(x) for x in group.columns.ravel()]
    return group

def group_to_feature(df, group, target_cols):
    df = df.merge(group, how='left', left_on='region', right_on='region_')
    df = df.drop('region_', axis=1)
    for col in target_cols:
        df['{0}_region_mean_diff'.format(col)] = df[col] - df['{0}_mean'.format(col)]
        df['{0}_region_mean_diff_region_std_diff'.format(col)] = df['{0}_region_mean_diff'.format(col)] - df['{0}_std'.format(col)]    
        df['{0}_region_mean_div_kurtosis'.format(col)] = df['{0}_region_mean_diff'.format(col)] / df['{0}_kurtosis'.format(col)]
        df['{0}_region_mean_div_skew'.format(col)] = df['{0}_region_mean_diff'.format(col)] / df['{0}_skew'.format(col)]
    
    return df

def require_median_dict(train_df, test_df):
    df = pd.concat([train_df, test_df])
    cols = df.columns[df.isna().sum() > 0]
    return {col: df[col].median() for col in cols}

def correlation_to_pca(train_df, test_df):
    all_df = pd.concat([train_df[test_df.columns], test_df])
    all_df = all_df.reset_index(drop=True)
    train_fill_df, test_fill_df = train_df.copy(), test_df.copy()
    for col in all_df.columns[all_df.isna().sum() > 0].values:
        all_df[col] = all_df[col].fillna(value=all_df[col].median())
        train_fill_df[col] = train_fill_df[col].fillna(value=all_df[col].median())
        test_fill_df[col] = test_fill_df[col].fillna(value=all_df[col].median())
    
    pca = PCA(n_components=1, random_state=0)
    pca.fit(all_df[['loudness', 'acousticness']])
    train_df['pca_loudness_acousticness'] = pca.transform(train_fill_df[['loudness', 'acousticness']])
    test_df['pca_loudness_acousticness'] = pca.transform(test_fill_df[['loudness', 'acousticness']])
    
    pca = PCA(n_components=1, random_state=0)
    pca.fit(all_df[['energy', 'acousticness']])
    train_df['pca_energy_acousticness'] = pca.transform(train_fill_df[['energy', 'acousticness']])
    test_df['pca_energy_acousticness'] = pca.transform(test_fill_df[['energy', 'acousticness']])
    
    pca = PCA(n_components=1, random_state=0)
    pca.fit(all_df[['danceability', 'positiveness']])
    train_df['pca_danceability_positiveness'] = pca.transform(train_fill_df[['danceability', 'positiveness']])
    test_df['pca_danceability_positiveness'] = pca.transform(test_fill_df[['danceability', 'positiveness']])
    
    pca = PCA(n_components=1, random_state=0)
    pca.fit(all_df[['energy', 'loudness']])
    train_df['pca_energy_loudness'] = pca.transform(train_fill_df[['energy', 'loudness']])
    test_df['pca_energy_loudness'] = pca.transform(test_fill_df[['energy', 'loudness']])
    
    return train_df, test_df