// #include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__device__ void compute_pair(int i, int j, vector3* pos, double* mass, vector3 out) {
    if (i == j) {
        out[0] = out[1] = out[2] = 0;
        return;
    }

    vector3 distance;
    for (int k = 0; k < 3; k++)
        distance[k] = pos[i][k] - pos[j][k];

    double mag_sq =
        distance[0]*distance[0] +
        distance[1]*distance[1] +
        distance[2]*distance[2];

    double mag = sqrt(mag_sq);

    double accelmag = -GRAV_CONSTANT * mass[j] / mag_sq;

    out[0] = accelmag * distance[0] / mag;
    out[1] = accelmag * distance[1] / mag;
    out[2] = accelmag * distance[2] / mag;
}

__global__ void accel_kernel(vector3* pos, double* mass, vector3* accels, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        compute_pair(i, j, pos, mass, accels[i*N + j]);
    }
}

__global__ void integrate_kernel(vector3* pos, vector3* vel, vector3* accels, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    vector3 accel_sum = {0,0,0};

    for (int j = 0; j < N; j++) {
        accel_sum[0] += accels[(i*N + j)*3][0];
        accel_sum[1] += accels[(i*N + j)*3][1];
        accel_sum[2] += accels[(i*N + j)*3][2];
    }

    for (int k = 0; k < 3; k++) {
        vel[i][k] += accel_sum[k] * INTERVAL;
        pos[i][k] += vel[i][k] * INTERVAL;
    }
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

    accel_kernel<<<grid, block>>>(dPos, dMass, dAccels, N);

    // Integrate kernel
    int threads = 256;
    integrate_kernel<<<(N+threads-1)/threads, threads>>>(dPos, dVel, dAccels, N);

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