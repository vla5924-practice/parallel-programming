// Copyright 2020 Vlasov Maksim
#include <mpi.h>
#include "broadcast.h"

inline int choose_rank(int rank, int root) {
    if (rank == root)
        return 0;
    if (rank == 0)
        return root;
    return rank;
}

int broadcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank != root) {
        MPI_Status status;
        int s_rank = (rank % 2 != 0) ? (rank - 1) / 2 : (rank - 2) / 2;
        MPI_Recv(buffer, count, datatype, choose_rank(s_rank, root), 0, comm, &status);
    }
    int r_rank_1 = 2 * rank + 1;
    int r_rank_2 = 2 * rank + 2;
    if (r_rank_1 < size) {
        MPI_Send(buffer, count, datatype, choose_rank(r_rank_1, root), 0, comm);
        if (r_rank_2 < size) {
            MPI_Send(buffer, count, datatype, choose_rank(r_rank_2, root), 0, comm);
        }
    }
    return MPI_SUCCESS;
}
