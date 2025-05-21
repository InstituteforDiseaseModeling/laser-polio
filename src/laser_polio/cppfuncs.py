import ctypes
from pathlib import Path

import numpy as np

# Load the shared library (adjust the path as needed)
lib_path = Path(__file__).parent / "compiled.cpython-312-darwin.so"
lib = ctypes.CDLL(lib_path)

# Define argument and return types for census
lib.census.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # node_ids
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # disease_state
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # paralyzed
    ctypes.c_int32,  # num_nodes
    ctypes.c_int32,  # num_people
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # pS
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # pE
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # pI
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # pR
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # pP
]
lib.census.restype = None


def census(num_nodes, num_people, node_ids, disease_state, paralyzed, pS, pE, pI, pR, pP):
    lib.census(
        node_ids,
        disease_state,
        paralyzed,
        np.int32(num_nodes),
        np.int32(num_people),
        pS,
        pE,
        pI,
        pR,
        pP,
    )

    return


# Define argument and return types for set_omp_seeds
lib.set_seeds.argtypes = [ctypes.c_int32]  # seed
lib.set_seeds.restype = None


def set_omp_seeds(seed):
    lib.set_seeds(np.int32(seed))

    return


# Define arguments and return types for disease_state_step
lib.disease_state_step.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # disease_state
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # exposure_timer
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # infection_timer
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # paralyzed
    ctypes.c_float,  # p_paralysis
    ctypes.c_int32,  # num_people
]
lib.disease_state_step.restype = None


def disease_state_step(disease_state, exposure_timer, infection_timer, paralyzed, p_paralysis, num_people):
    lib.disease_state_step(disease_state, exposure_timer, infection_timer, paralyzed, np.float32(p_paralysis), np.int32(num_people))

    return


lib.tx_step_part_one.argtypes = [
    ctypes.c_int32,  # num_nodes
    ctypes.c_int32,  # num_people
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # disease_state
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # node_ids
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # risks
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # infectivity
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # network
    ctypes.c_float,  # seasonality
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # r0_scalars
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # alive_by_node
    # outputs
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # beta_by_node
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # base_prob_inf
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # exposure_by_node
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # new_infections
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # sus_by_node
]
lib.tx_step_part_one.restype = None


def tx_step_part_one(
    num_nodes,
    num_people,
    disease_state,
    node_ids,
    risks,
    infectivity,
    network,
    seasonality,
    r0_scalars,
    alive_by_node,
    beta_by_node,
    base_prob_inf,
    exposure_by_node,
    new_infections,
    sus_by_node,
):
    lib.tx_step_part_one(
        np.int32(num_nodes),
        np.int32(num_people),
        disease_state,
        node_ids,
        risks,
        infectivity,
        network,
        np.float32(seasonality),
        r0_scalars,
        alive_by_node,
        beta_by_node,
        base_prob_inf,
        exposure_by_node,
        new_infections,
        sus_by_node,
    )

    return


lib.get_threadid.argtypes = [
    ctypes.c_int32,  # num
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # out`
]
lib.get_threadid.restype = None


def get_threadid(num, out):
    lib.get_threadid(np.int32(num), out)

    return


lib.get_uniform.argtypes = [
    ctypes.c_int32,  # num
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # out
]
lib.get_uniform.restype = None


def get_uniform(num, out):
    lib.get_uniform(np.int32(num), out)

    return


if __name__ == "__main__":
    num_nodes = 100
    num_people = 100_000
    node_ids = np.random.randint(0, num_nodes, size=num_people, dtype=np.int32)
    disease_state = np.random.randint(0, 5, size=num_people, dtype=np.int32) - 1
    paralyzed = np.random.randint(0, 2, size=num_people, dtype=np.int32)
    pS = np.zeros(num_nodes, dtype=np.int32)
    pE = np.zeros(num_nodes, dtype=np.int32)
    pI = np.zeros(num_nodes, dtype=np.int32)
    pR = np.zeros(num_nodes, dtype=np.int32)
    pP = np.zeros(num_nodes, dtype=np.int32)
    print("Running census function...", end="")
    census(num_nodes, num_people, node_ids, disease_state, paralyzed, pS, pE, pI, pR, pP)

    assert np.all(pS == np.bincount(node_ids[np.where(disease_state == 0)[0]]))
    assert np.all(pE == np.bincount(node_ids[np.where(disease_state == 1)[0]]))
    assert np.all(pI == np.bincount(node_ids[np.where(disease_state == 2)[0]]))
    assert np.all(pR == np.bincount(node_ids[np.where(disease_state == 3)[0]]))
    assert np.all(pP == np.bincount(node_ids[np.where((disease_state >= 0) & (paralyzed > 0))[0]]))
    print("Census function executed successfully.")

    exposure_timer = np.zeros(num_people, dtype=np.int32)
    exposure_timer[np.where(disease_state == 1)[0]] = np.random.randint(0, 7, size=np.sum(disease_state == 1), dtype=np.int32) - 2
    infection_timer = np.zeros(num_people, dtype=np.int32)
    infection_timer[np.where(disease_state == 2)[0]] = np.random.randint(0, 7, size=np.sum(disease_state == 2), dtype=np.int32) - 2

    exposure_expiring = np.where((disease_state == 1) & (exposure_timer <= 0))[0]
    infection_expiring = np.where((disease_state == 2) & (infection_timer <= 0))[0]

    first_time = np.zeros(num_nodes, dtype=np.int32)
    get_threadid(num_nodes, first_time)
    second_time = np.zeros(num_nodes, dtype=np.int32)
    get_threadid(num_nodes, second_time)
    assert np.all(first_time == second_time)
    print("Thread ID generation is reproducible.")

    first_draw = np.zeros(num_nodes, dtype=np.float32)
    set_omp_seeds(20250520)
    get_uniform(num_nodes, first_draw)
    second_draw = np.zeros(num_nodes, dtype=np.float32)
    set_omp_seeds(20250520)
    get_uniform(num_nodes, second_draw)
    assert np.all(first_draw == second_draw)
    print("Uniform random number generation is reproducible.")

    print("Setting OpenMP seeds...", end="")
    set_omp_seeds(20250520)
    print("set seeds function executed successfully.")

    ds_copy = np.array(disease_state, copy=True)
    et_copy = np.array(exposure_timer, copy=True)
    it_copy = np.array(infection_timer, copy=True)
    paralyzed[:] = 0
    par_copy = np.array(paralyzed, copy=True)

    print("Running disease_state_step...", end="")
    disease_state_step(
        disease_state,
        exposure_timer,
        infection_timer,
        paralyzed,
        np.float32(1.0 / 1000.0),
        np.int32(num_people),
    )

    assert np.all(disease_state[exposure_expiring] == 2)
    assert np.all(disease_state[infection_expiring] == 3)
    print("Disease state step function executed successfully.")

    print("Checking for reproducibility...", end="")
    set_omp_seeds(20250520)
    disease_state_step(
        ds_copy,
        et_copy,
        it_copy,
        par_copy,
        np.float32(1.0 / 1000.0),
        np.int32(num_people),
    )
    assert np.all(ds_copy == disease_state)
    assert np.all(et_copy == exposure_timer)
    assert np.all(it_copy == infection_timer)
    assert np.all(par_copy == paralyzed)
    print("Reproducibility check passed.")

    # additional inputs
    risks = np.random.rand(num_people).astype(np.float32)
    infectivity = 0.8 + 0.4 * np.random.rand(num_people).astype(np.float32)
    network = 0.05 * np.random.rand(num_nodes * num_nodes).astype(np.float64).reshape(num_nodes, num_nodes)
    seasonality = 0.9 + 0.2 * np.random.rand()
    r0_scalars = 0.5 + np.random.rand(num_nodes)
    alive_by_node = np.bincount(node_ids[np.where(disease_state >= 0)[0]]).astype(np.int32)
    # outputs
    beta_by_node = np.empty(num_nodes, dtype=np.float64)
    base_prob_inf = np.empty(num_nodes, dtype=np.float64)
    exposure_by_node = np.empty(num_nodes, dtype=np.float64)
    new_infections = np.empty(num_nodes, dtype=np.int32)
    sus_by_node = np.empty(num_nodes, dtype=np.int32)
    tx_step_part_one(
        num_nodes,
        num_people,
        disease_state,
        node_ids,
        risks,
        infectivity,
        network,
        seasonality,
        r0_scalars,
        alive_by_node,
        beta_by_node,
        base_prob_inf,
        exposure_by_node,
        new_infections,
        sus_by_node,
    )

    sus_indices = np.where(disease_state == 0)[0]
    sbn = np.bincount(node_ids[sus_indices], minlength=num_nodes)
    ebn = np.bincount(node_ids[sus_indices], weights=risks[sus_indices], minlength=num_nodes)
    inf_indices = np.where(disease_state == 2)[0]
    bbn = np.bincount(node_ids[inf_indices], weights=infectivity[inf_indices], minlength=num_nodes)

    tx = bbn * network
    bbn += tx.sum(axis=1) - tx.sum(axis=0)
    bbn = np.maximum(bbn, 0.0)
    bbn = bbn * seasonality * r0_scalars

    pai = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        if alive_by_node[i] > 0:
            pai[i] = bbn[i] / alive_by_node[i]
    bpi = 1 - np.exp(-pai)
    ebn *= bpi

    new_inf = np.random.poisson(ebn)

    assert np.allclose(bbn, beta_by_node)
    assert np.allclose(bpi, base_prob_inf)
    assert np.allclose(ebn, exposure_by_node)
    assert np.all(sbn == sus_by_node)
    print(f"{new_infections.sum()=}, {new_inf.sum()=}")

    print("done")
