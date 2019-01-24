import pandas as pd
import numpy as np

def load_targets():
    df_target_playlists = pd.read_csv('target_playlists.csv')
    targets = np.array(list(filter(lambda x: x < np.inf, np.array(df_target_playlists))))
    sorted_target_playlists = np.sort(targets.ravel())
    return sorted_target_playlists
    
def load_recs():
    df_recs = pd.read_csv('ok1.csv')
    return np.array(df_recs)

def save_recommendations(target_indices, recommendations_list):
    data_out = {'playlist_id': target_indices, 'track_ids': recommendations_list}
    df_out = pd.DataFrame(data=data_out)
    df_out.to_csv("out.csv", index=False)

targets = load_targets()
recs = load_recs()

recs = recs[targets]
recs = recs[:,1]

save_recommendations(targets, recs)
