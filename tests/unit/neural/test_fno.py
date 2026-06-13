import numpy as np
from astra.neural.fno import PorkchopFNO, INPUT_DIM

def test_fourier_encoding_shape():
    fno = PorkchopFNO(n_fourier_features=32, hidden_dim=64)
    x = np.random.randn(10, INPUT_DIM).astype(np.float32)
    out = fno.forward(x)
    assert out.shape == (10, 1)
    assert np.all(out > 0.0)  # Softplus always positive

def test_prediction_clipped():
    fno = PorkchopFNO(n_fourier_features=32, hidden_dim=64)
    x = np.zeros((5, INPUT_DIM), dtype=np.float32)
    out = fno.forward(x)
    assert np.all(out >= 0.1) and np.all(out <= 50.0)

def test_surrogate_interface():
    fno = PorkchopFNO(n_fourier_features=32, hidden_dim=64)
    feat = np.zeros(INPUT_DIM, dtype=np.float32)
    out = fno.predict(feat)
    assert out.requires_physics_validation is True
    assert out.prediction > 0.0

def test_training_loss_decreases():
    fno = PorkchopFNO(n_fourier_features=32, hidden_dim=64, lr=1e-2, seed=0)
    
    dep_epochs = np.linspace(0, 1e7, 10)
    tof_arr = np.linspace(1e6, 5e7, 10)
    dv_grid = np.full((10, 10), 5.0)

    dep_states = {float(d): (np.array([1e8, 0, 0]), np.array([0, 30, 0]))
                  for d in dep_epochs}
    arr_states = {}
    for d in dep_epochs:
        for t in tof_arr:
            arr_states[float(d + t)] = (np.array([2e8, 0, 0]), np.array([0, 24, 0]))

    result = fno.train_on_grid(
        dep_epochs, tof_arr, dv_grid,
        float(dep_epochs[0]), float(dep_epochs[-1]),
        float(tof_arr[0]), float(tof_arr[-1]),
        dep_states, arr_states, 0.0,
        epochs=10, batch_size=32,
    )
    assert "train_loss" in result
    assert len(result["train_loss"]) == 10
    assert fno.is_trained()

def test_find_top_k_candidates():
    fno = PorkchopFNO()
    grid = np.array([[5.0, 3.0, 7.0], [4.0, 2.0, 6.0]], dtype=np.float32)
    deps = np.array([0.0, 1e6])
    tofs = np.array([1e6, 2e6, 3e6])
    cands = fno.find_top_k_candidates(grid, deps, tofs, top_k=2)
    assert len(cands) == 2
    assert cands[0][0] == 1e6 and cands[0][1] == 2e6  # min is grid[1,1]=2.0
