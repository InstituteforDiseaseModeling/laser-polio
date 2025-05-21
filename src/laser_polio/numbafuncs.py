import numba as nb
import numpy as np


@nb.njit(parallel=True)
def step_nb(disease_state, exposure_timer, infection_timer, acq_risk_multiplier, daily_infectivity, paralyzed, p_paralysis, active_count):
    for i in nb.prange(active_count):
        if disease_state[i] == 1:  # Exposed
            if exposure_timer[i] <= 0:
                disease_state[i] = 2  # Become infected
                # Apply paralysis probability immediately after infection
                if np.random.random() < p_paralysis:
                    paralyzed[i] = 1
            exposure_timer[i] -= 1  # Decrement exposure timer so that they become infected on the next timestep

        if disease_state[i] == 2:  # Infected
            if infection_timer[i] <= 0:
                disease_state[i] = 3  # Become recovered
                # acq_risk_multiplier[i] = 0.0  # Reset risk
                # daily_infectivity[i] = 0.0  # Reset infectivity
            infection_timer[i] -= 1  # Decrement infection timer so that they recover on the next timestep

    return


@nb.njit((nb.int32[:], nb.int32[:], nb.int32[:], nb.int32, nb.int32), nogil=True)  # , cache=True)
def count_SEIRP(node_id, disease_state, paralyzed, n_nodes, n_people):
    """
    Go through each person exactly once and increment counters for their node.

    node_id:        array of node IDs for each individual
    disease_state:  array storing each person's disease state (-1=dead/inactive, 0=S, 1=E, 2=I, 3=R)
    paralyzed:      array (0 or 1) if the person is paralyzed
    n_nodes:        total number of nodes

    Returns: S, E, I, R, P arrays, each length n_nodes
    """

    n_threads = nb.get_num_threads()
    # S = np.zeros((n_threads, n_nodes), dtype=np.int32)
    # E = np.zeros((n_threads, n_nodes), dtype=np.int32)
    # I = np.zeros((n_threads, n_nodes), dtype=np.int32)
    # R = np.zeros((n_threads, n_nodes), dtype=np.int32)
    SEIR = np.zeros((n_threads, n_nodes, 4), dtype=np.int32)  # S, E, I, R
    P = np.zeros((n_threads, n_nodes), dtype=np.int32)

    # Single pass over the entire population
    for i in nb.prange(n_people):
        if disease_state[i] >= 0:  # Only count those who are alive
            nd = node_id[i]
            ds = disease_state[i]

            tid = nb.get_thread_id()
            # if ds == 0:  # Susceptible
            #     S[tid, nd] += 1
            # elif ds == 1:  # Exposed
            #     E[tid, nd] += 1
            # elif ds == 2:  # Infected
            #     I[tid, nd] += 1
            # elif ds == 3:  # Recovered
            #     R[tid, nd] += 1
            # NOTE: This only works if disease_state is contiguous, 0..N
            SEIR[tid, nd, ds] += 1

            # Check paralyzed
            if paralyzed[i] == 1:
                P[tid, nd] += 1

    # return S, E, I, R, P
    return SEIR[:, :, 0].sum(axis=0), SEIR[:, :, 1].sum(axis=0), SEIR[:, :, 2].sum(axis=0), SEIR[:, :, 3].sum(axis=0), P.sum(axis=0)
