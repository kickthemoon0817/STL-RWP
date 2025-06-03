import pathlib
import numpy as np


data_path = pathlib.Path("./data")

u10n_arr = np.load(data_path / "u10n_array.npy")            # (T, 40, 40)
v10n_arr = np.load(data_path / "v10n_array.npy")            # (T, 40, 40)
swh_arr = np.load(data_path / "swh_interp_array.npy")       # (T, 40, 40)
times = np.load(data_path / "times.npy")                    # (T,)

T, H, W = u10n_arr.shape
assert v10n_arr.shape == (T, H, W)
assert swh_arr.shape == (T, H, W)
assert times.shape == (T,)

K = 6
M = 1
N_train_val = 35064


u10n_tv = u10n_arr[:N_train_val]    # (35064, 40, 40)
v10n_tv = v10n_arr[:N_train_val]
swh_tv = swh_arr[:N_train_val]

u10n_test = u10n_arr[N_train_val:]  # (T - 35064, 40, 40)
v10n_test = v10n_arr[N_train_val:]
swh_test = swh_arr[N_train_val:]

times_tv = times[:N_train_val]
times_test = times[N_train_val:]

num_samples_tv = N_train_val - K - M + 1  # 35058

X_tv = np.zeros((num_samples_tv, K, 3, H, W), dtype=u10n_tv.dtype)
Y_tv = np.zeros((num_samples_tv, 3, H, W), dtype=u10n_tv.dtype)

for i in range(num_samples_tv):
    u_block = u10n_tv[i : i + K]   # (6, 40, 40)
    v_block = v10n_tv[i : i + K]
    s_block = swh_tv[i : i + K]
    X_tv[i] = np.stack([u_block, v_block, s_block], axis=1)

    u_target = u10n_tv[i + K]      # (40, 40)
    v_target = v10n_tv[i + K]
    s_target = swh_tv[i + K]
    Y_tv[i] = np.stack([u_target, v_target, s_target], axis=0)

print("X_tv.shape =", X_tv.shape)  # (35058, 6, 3, 40, 40)
print("Y_tv.shape =", Y_tv.shape)  # (35058, 3, 40, 40)

np.random.seed(42)
perm = np.random.permutation(num_samples_tv)

X_tv_shuffled = X_tv[perm]
Y_tv_shuffled = Y_tv[perm]

n_train = 28046
n_val = 7012

X_train = X_tv_shuffled[:n_train]
Y_train = Y_tv_shuffled[:n_train]

X_val = X_tv_shuffled[n_train:n_train + n_val]
Y_val = Y_tv_shuffled[n_train:n_train + n_val]

print("X_train.shape =", X_train.shape)  # (28046, 6, 3, 40, 40)
print("Y_train.shape =", Y_train.shape)  # (28046, 3, 40, 40)
print("X_val.shape   =", X_val.shape)    # (7012, 6, 3, 40, 40)
print("Y_val.shape   =", Y_val.shape)    # (7012, 3, 40, 40)

N_test = u10n_test.shape[0]
num_samples_test = N_test - K - M + 1

X_test = np.zeros((num_samples_test, K, 3, H, W), dtype=u10n_test.dtype)
Y_test = np.zeros((num_samples_test, 3, H, W), dtype=u10n_test.dtype)

for j in range(num_samples_test):
    u_block = u10n_test[j : j + K]
    v_block = v10n_test[j : j + K]
    s_block = swh_test[j : j + K]
    X_test[j] = np.stack([u_block, v_block, s_block], axis=1)

    u_target = u10n_test[j + K]
    v_target = v10n_test[j + K]
    s_target = swh_test[j + K]
    Y_test[j] = np.stack([u_target, v_target, s_target], axis=0)

print("X_test.shape =", X_test.shape)  # (N_test - 6 - 1 + 1, 6, 3, 40, 40)
print("Y_test.shape =", Y_test.shape)  # (N_test - 6 - 1 + 1, 3, 40, 40)

np.save(data_path / "X_train.npy", X_train)
np.save(data_path / "Y_train.npy", Y_train)
np.save(data_path / "X_val.npy",   X_val)
np.save(data_path / "Y_val.npy",   Y_val)
np.save(data_path / "X_test.npy",  X_test)
np.save(data_path / "Y_test.npy",  Y_test)

print("Saved: X_train.npy, Y_train.npy, X_val.npy, Y_val.npy, X_test.npy, Y_test.npy")
