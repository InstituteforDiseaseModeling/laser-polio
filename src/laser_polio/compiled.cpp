#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <omp.h>

using gen_t = std::mt19937_64;
using pgen_t = gen_t*;
using uniform_t = std::uniform_real_distribution<float_t>;

pgen_t* pgenerators = nullptr;

extern "C" {
    
    void set_seeds(int32_t seed) {
        size_t num_threads = omp_get_max_threads();

        if (pgenerators == nullptr) {
            pgenerators = new pgen_t[num_threads];
        } else {
            for (size_t i = 0; i < num_threads; ++i) {
                delete pgenerators[i];
            }
        }

        for (size_t i = 0; i < num_threads; ++i) {
            pgenerators[i] = new gen_t(20250520 + i);
        }
    }

    void census(
        int32_t* node_ids,
        int32_t* disease_state,
        int32_t* paralyzed,
        int32_t num_nodes,
        int32_t num_people,
        int32_t* pS,
        int32_t* pE,
        int32_t* pI,
        int32_t* pR,
        int32_t* pP)
    {
        size_t num_threads = omp_get_max_threads();

        size_t num_state_counters_per_thread = num_nodes * 4;
        // "Round up" to the nearest multiple of 32 (128 bytes)
        num_state_counters_per_thread = ((num_state_counters_per_thread + 31) / 32) * 32;
        size_t num_state_counters = num_threads * num_state_counters_per_thread;
        
        int32_t* state_counts = new int32_t[num_state_counters];
        
        size_t num_state_bytes = sizeof(int32_t) * num_state_counters;
        memset(state_counts, 0, num_state_bytes);
        
        size_t num_paralyzed_counters_per_thread = num_nodes;
        // "Round up" to the nearest multiple of 32 (128 bytes)
        num_paralyzed_counters_per_thread = ((num_paralyzed_counters_per_thread + 31) / 32) * 32;
        size_t num_paralyzed_counters = num_threads * num_paralyzed_counters_per_thread;
        int32_t* paralyzed_counts = new int32_t[num_paralyzed_counters];
        size_t num_paralyzed_bytes = sizeof(int32_t) * num_paralyzed_counters;
        memset(paralyzed_counts, 0, num_paralyzed_bytes);

        #pragma omp parallel for
        for (int i = 0; i < num_people; ++i) {
            int32_t state = disease_state[i];
            if (state >= 0) {
                int32_t tid = omp_get_thread_num();
                int32_t nid = node_ids[i];
                size_t offset = tid * num_state_counters_per_thread + 4 * nid + state;
                state_counts[offset]++;
                if (paralyzed[i]) {
                    offset = tid * num_paralyzed_counters_per_thread + nid;
                    paralyzed_counts[offset]++;
                }
            }
        }

        memset(pS, 0, sizeof(int32_t) * num_nodes);
        memset(pE, 0, sizeof(int32_t) * num_nodes);
        memset(pI, 0, sizeof(int32_t) * num_nodes);
        memset(pR, 0, sizeof(int32_t) * num_nodes);
        memset(pP, 0, sizeof(int32_t) * num_nodes);

        // Aggregate results from all threads
        for (int inode = 0; inode < num_nodes; ++inode) {
            size_t state_offset = 4 * inode;
            size_t paralyzed_offset = inode;
            for (size_t ithread = 0; ithread < num_threads; ++ithread) {
                pS[inode] += state_counts[state_offset + 0];
                pE[inode] += state_counts[state_offset + 1];
                pI[inode] += state_counts[state_offset + 2];
                pR[inode] += state_counts[state_offset + 3];

                pP[inode] += paralyzed_counts[paralyzed_offset];

                state_offset += num_state_counters_per_thread;
                paralyzed_offset += num_paralyzed_counters_per_thread;
            }
        }

        // I like to free up in reverse order of allocation
        delete[] paralyzed_counts;
        delete[] state_counts;
    }

    void disease_state_step(
        int32_t* disease_state,
        int32_t* exposure_timer,
        int32_t* infection_timer,
        int32_t* paralyzed,
        float_t p_paralysis,
        int32_t num_people
        )
    {
        if (pgenerators == nullptr) {
            printf("Error: Random number generator not initialized.\n");
            return;
        }
        #pragma omp parallel for
        for (int i = 0; i < num_people; ++i) {
            int32_t state = disease_state[i];
            if (state == 1) {
                if (exposure_timer[i] <= 0) {
                    disease_state[i] = 2;
                    int tid = omp_get_thread_num();
                    auto pgenerator = (pgen_t)pgenerators[tid];
                    float_t draw = uniform_t(0.0, 0.1)(*pgenerator);
                    if (draw < p_paralysis) {
                        paralyzed[i] = 1;
                    }
                } else {
                    exposure_timer[i] -= 1;
                }
            } else if (state == 2) {
                if (infection_timer[i] <= 0) {
                    disease_state[i] = 3;
                } else {
                    infection_timer[i] -= 1;
                }
            }
        }
    }

    void tx_step_part_one(
        int32_t num_nodes,
        int32_t num_people,
        int32_t* disease_state,
        int32_t* node_ids,
        float_t* risks,
        float_t* infectivity,
        double_t* network,
        float_t seasonality,
        double_t* r0_scalars,
        int32_t* alive_by_node,
        // outputs
        double_t* beta_by_node,
        double_t* base_prob_inf,
        double_t* exposure_by_node,
        int32_t* new_infections,
        int32_t* sus_by_node
    )
    {
        size_t num_threads = omp_get_max_threads();
        double_t* tl_beta_by_node = new double_t[num_threads * num_nodes];
        double_t* tl_exposure_by_node = new double_t[num_threads * num_nodes];
        int32_t* tl_sus_by_node = new int32_t[num_threads * num_nodes];

        memset(tl_beta_by_node, 0, sizeof(double_t) * num_threads * num_nodes);
        memset(tl_exposure_by_node, 0, sizeof(double_t) * num_threads * num_nodes);
        memset(tl_sus_by_node, 0, sizeof(int32_t) * num_threads * num_nodes);

        #pragma omp parallel for
        for (int i = 0; i < num_people; ++i) {
            int32_t state = disease_state[i];
            if (state == 0) {
                int32_t tid = omp_get_thread_num();
                int32_t nid = node_ids[i];
                tl_exposure_by_node[tid * num_nodes + nid] += risks[i];
                ++tl_sus_by_node[tid * num_nodes + nid];
            } else if (state == 2) {
                int32_t tid = omp_get_thread_num();
                int32_t nid = node_ids[i];
                tl_beta_by_node[tid * num_nodes + nid] += infectivity[i];
            }
        }

        memset(beta_by_node, 0, sizeof(double_t) * num_nodes);
        memset(exposure_by_node, 0, sizeof(double_t) * num_nodes);
        memset(sus_by_node, 0, sizeof(int32_t) * num_nodes);

        // Aggregate results from all threads
        // TODO - consider using OpenMP reduction clause
        #pragma omp parallel for
        for (int nid = 0; nid < num_nodes; ++nid) {
            size_t offset = nid;
            for (size_t tid = 0; tid < num_threads; ++tid) {
                beta_by_node[nid] += tl_beta_by_node[offset];
                exposure_by_node[nid] += tl_exposure_by_node[offset];
                sus_by_node[nid] += tl_sus_by_node[offset];

                offset += num_nodes;
            }
        }

        // I like to free up in reverse order of allocation
        delete[] tl_sus_by_node;
        delete[] tl_exposure_by_node;
        delete[] tl_beta_by_node;

        double_t* incoming = new double_t[num_nodes];
        double_t* outgoing = new double_t[num_nodes];
        memset(incoming, 0, sizeof(double_t) * num_nodes);
        memset(outgoing, 0, sizeof(double_t) * num_nodes);
        for (int32_t i = 0; i < num_nodes; ++i) {
            for (int32_t j = 0; j < num_nodes; ++j) {
                double_t flow = beta_by_node[j] * network[i * num_nodes + j];
                incoming[i] += flow;
                outgoing[j] += flow;
            }
        }
        for (int32_t k = 0; k < num_nodes; ++k) {
            beta_by_node[k] += incoming[k] - outgoing[k];
        }
        delete[] outgoing;
        delete[] incoming;

        #pragma omp parallel for
        for (int32_t i = 0; i < num_nodes; ++i) {
            double_t beta = std::max(0.0, beta_by_node[i]) * seasonality * r0_scalars[i];
            beta_by_node[i] = beta;
            int32_t alive = alive_by_node[i];
            // double_t per_agent_inf_rate = (alive > 0) ? beta / alive : 0.0;
            double_t per_agent_inf_rate = (alive > 0) ? beta / alive : beta;
            base_prob_inf[i] = 1.0 - exp(-per_agent_inf_rate);
            exposure_by_node[i] *= base_prob_inf[i];
            int32_t tid = omp_get_thread_num();
            new_infections[i] = std::poisson_distribution<int32_t>(exposure_by_node[i])(*pgenerators[tid]);
        }
    }

    void get_threadid(int32_t num, int32_t* out) {
        #pragma omp parallel for
        for (int i = 0; i < num; ++i) {
            out[i] = omp_get_thread_num();
        }
    }

    void get_uniform(int32_t num, float_t* out) {
        if (pgenerators == nullptr) {
            printf("Error: Random number generator not initialized.\n");
            return;
        }
        #pragma omp parallel for
        for (int i = 0; i < num; ++i) {
            int tid = omp_get_thread_num();
            auto pgenerator = (pgen_t)pgenerators[tid];
            out[i] = uniform_t(0.0, 1.0)(*pgenerator);
        }
    }

}
