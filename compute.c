#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

// Cuda version of first compute the pairwise accelerations.  Effect is on the first argument.
__global__ void accel_kernel(vector3* accels, vector3* pos, double* mass, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;

    if (i == j) {
        accels[i * N + j][0] = 0.0;
        accels[i * N + j][1] = 0.0;
        accels[i * N + j][2] = 0.0;
        return;
    }

    double dx = pos[i][0] - pos[j][0];
    double dy = pos[i][1] - pos[j][1];
    double dz = pos[i][2] - pos[j][2];

    double dist_sq = dx*dx + dy*dy + dz*dz;
    double dist = sqrt(dist_sq);

    if (dist_sq == 0.0) return; // incase of overlap

    double accelmag = -GRAV_CONSTANT * mass[j] / dist_sq;

    accels[i * N + j][0] = accelmag * dx / dist;
    accels[i * N + j][1] = accelmag * dy / dist;
    accels[i * N + j][2] = accelmag * dz / dist;
}

__global__ void integrate_kernel(vector3* accels, vector3* pos, vector3* vel, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (int j = 0; j < N; j++) {
        ax += accels[i * N + j][0];
        ay += accels[i * N + j][1];
        az += accels[i * N + j][2];
    }

    vel[i][0] += ax * INTERVAL;
    vel[i][1] += ay * INTERVAL;
    vel[i][2] += az * INTERVAL;

    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

void compute() {
    int N = NUMENTITIES;

    size_t vecSize = sizeof(vector3) * N;
    size_t fullVec = sizeof(vector3) * N * N;
    size_t massSize = sizeof(double) * N;

    // Flatten host data
    vector3 *hPosFlat = (vector3*)malloc(sizeof(vector3)*N);
    vector3 *hVelFlat = (vector3*)malloc(sizeof(vector3)*N);

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++) {
            hPosFlat[i][k] = hPos[i][k];
            hVelFlat[i][k] = hVel[i][k];
        }
    }

    // GPU memory
    vector3 *dPos, *dVel, *dAccels;
    double *dMass;

    cudaMalloc(&dPos, vecSize);
    cudaMalloc(&dVel, vecSize);
    cudaMalloc(&dAccels, fullVec);
    cudaMalloc(&dMass, massSize);

    cudaMemcpy(dPos, hPosFlat, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVelFlat, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dMass, mass, massSize, cudaMemcpyHostToDevice);

    // Launch accel kernel (2D grid)
    dim3 block(16,16);
    dim3 grid((N+15)/16, (N+15)/16);

    accel_kernel<<<grid, block>>>(dAccels, dPos, dMass, N);
    cudaDeviceSynchronize();

    // Integrate kernel
    int threads = 256;
    integrate_kernel<<<(N+threads-1)/threads, threads>>>(dAccels, dPos, dVel, N);

    cudaMemcpy(hPosFlat, dPos, vecSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVelFlat, dVel, vecSize, cudaMemcpyDeviceToHost);

    // unpack back
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++) {
            hPos[i][k] = hPosFlat[i][k];
            hVel[i][k] = hVelFlat[i][k];
        }
    }

    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dAccels);
    cudaFree(dMass);

    free(hPosFlat);
    free(hVelFlat);
}