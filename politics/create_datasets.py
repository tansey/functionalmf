import numpy as np

np.random.seed(42)

df = np.load('gdelt.npz')

G20 = [(0, 'United States'),
        (1, 'Russian Federation'),
        (2, 'China'),
        (4, 'Japan'),
        (6, 'United Kingdom'),
        (8, 'South Korea'),
        (9, 'India'),
        (10, 'Turkey'),
        (11, 'France'),
        (16, 'Germany'),
        (18, 'Australia'),
        (25, 'Indonesia'),
        (28, 'Italy'),
        (31, 'Saudi Arabia'),
        (32, 'South Africa'),
        (34, 'Brazil'),
        (38, 'Mexico'),
        (44, 'Canada'),
        (48, 'Argentina')]
G20_indices = np.array([x[0] for x in G20])
G20_names = np.array([x[1] for x in G20])
dates = np.array([str(x.decode('UTF-8')) for x in df['dates']])
Y = df['Y']
actors = np.array([str(x.decode('UTF-8')) for x in df['actors']])
actions = np.array([str(x.decode('UTF-8')) for x in df['actions']])

# Filter down to the G20
Y = Y[G20_indices][:,G20_indices]
actors = actors[G20_indices]

# Pick one thing
action_idx = 2 # "Intend to Cooperate"
print(actions[action_idx])
Y = Y[:,:,action_idx].astype(float)

# # Hold out some years
# years = np.array([int(x[:4]) for x in dates])
# sender_idx, receiver_idx, year_start, year_end = [], [], [], []
# for yr in range(years.min(), years.max()+1):
#     yr_start = np.arange(years.shape[0])[years == yr][0]
#     yr_end = np.arange(years.shape[0])[years == yr][-1]+1
#     for i in range(Y.shape[0]):
#         for j in range(Y.shape[1]):
#             if i == j:
#                 # Nations don't send messages to themselves
#                 Y[i,j] = np.nan
#                 continue
#             sender_idx.append(i)
#             receiver_idx.append(j)
#             year_start.append(yr_start)
#             year_end.append(yr_end)
# indices = np.array([sender_idx, receiver_idx, year_start, year_end]).T
# to_hold = indices[np.random.choice(indices.shape[0], replace=False, size=int(np.ceil(indices.shape[0]*0.1)))]
# Y_train = np.copy(Y)
# for i,j,k,l in to_hold:
#     Y_train[i,j,k:l] = np.nan

# Hold out entire nation-nation pairs
indices = np.array([np.repeat(np.arange(Y.shape[0]), Y.shape[1]), np.tile(np.arange(Y.shape[0]), Y.shape[1])]).T
to_hold = indices[np.random.choice(indices.shape[0], replace=False, size=int(np.ceil(Y.shape[0]*Y.shape[1]*0.1)))]
Y_train = np.copy(Y)
for i,j in to_hold:
    Y_train[i,j] = np.nan

print('Held out {} nation pairs total'.format(to_hold.shape[0]))
# Save to file
np.save('cooperate', Y)
np.save('cooperate_train', Y_train)
# np.save('held_out_years', to_hold)
np.save('held_out', to_hold)
np.save('dates', dates)
np.save('nations', G20_names)









