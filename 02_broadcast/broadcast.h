// Copyright 2020 Vlasov Maksim
#pragma once

/**
 * Performs the same action as MPI_Bcast does
 * 
 * There are some comments on my implementation of the algorithm:
 * 
 * Binary heap is used in this algorithm. Each process send data to two others.
 * For clear implementation, there is a small hack: the rank of the root process 
 * is replaced with 0, and vice versa, due to what we really can interpret a set
 * of processes as a heap.
 * 
 * Let's say the process rank is x. Then, processes with odd ranks receive data 
 * from process with rank (x-1)/2 and processes with even ranks receive data from
 * process with rank (x-2)/2. Also, each process should send data to two processes
 * with ranks 2x+1 and 2x+2 if possible.
 */
int broadcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
