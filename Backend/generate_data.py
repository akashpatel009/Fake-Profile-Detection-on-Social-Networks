import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

data = pd.DataFrame({
    'has_profile_pic': np.random.choice([0, 1], n),
    'username_length': np.random.randint(5, 20, n),
    'num_digits_in_username': np.random.randint(0, 5, n),
    'bio_length': np.random.randint(0, 100, n),
    'follower_following_ratio': np.round(np.random.uniform(0.01, 2.0, n), 2),
    'is_fake': np.random.choice([0, 1], n, p=[0.7, 0.3])
})

data.to_csv('fake_profiles.csv', index=False)
print("✅ Dataset saved as fake_profiles.csv")