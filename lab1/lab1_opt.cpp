#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;


int main() {
    srand(42);
    omp_set_num_threads(24);   // 讓 OMP 自動吃滿核心
    std::cout << "OMP max threads: " << omp_get_max_threads() << std::endl;
    double start_time = omp_get_wtime();

    // init
    // ------------------------
    const int numParticles = 10000000;
    const int gridRows = 50000;
    const int gridCols = 50000;

    // main array：velocity, pressure, energy
    vector<double> velocity(numParticles), pressure(numParticles), energy(numParticles);

    // init velocity and pressure
    #pragma omp parallel for schedule(static)
        for (int i = 0; i < numParticles; ++i) {
            velocity[i] = i * 1.0;
            pressure[i] = (numParticles - i) * 1.0;
            energy[i] = velocity[i] + pressure[i];
        }

    double t1 = omp_get_wtime();
    cout << "init time: " << t1 - start_time << endl;

    // #pragma omp parallel for schedule(static)
    //     for (int i = 0; i < numParticles; ++i)
    //         energy[i] = velocity[i] + pressure[i];
    // double t1 = omp_get_wtime();
    // cout << "cal energy time: " << t1 - t0 << endl;

    double sinAccum[11] = {0};               // 0 不用  
    for (int L = 10; L <= 100; L += 10) {
        double acc = 0.0;
        for (int j = 0; j < L; ++j) acc += sin(j * 0.01);
        sinAccum[L / 10] = acc;              // index 1~10
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < numParticles; ++i) {
        int loops = (i % 10) * 10 + 10;      // 10~100
        double work = sinAccum[loops / 10];  // O(1) 查表
        velocity[i] = sin(energy[i]) + log1p(fabs(work));
    }
    double t2 = omp_get_wtime();
    cout << "update V time: " << t2 - t1 << endl;
        

    double fieldSum = 0;
    double A = 0.0, B = 0.0;

    #pragma omp parallel for reduction(+:A)
    for (int r = 0; r < gridRows; ++r)
        A += sqrt(r * 2.0);

    #pragma omp parallel for reduction(+:B)
    for (int c = 0; c < gridCols; ++c)
        B += log1p(c * 2.0);

    fieldSum = gridCols * A + gridRows * B;       // O(R+C) 而非 O(R·C)

    double t3 = omp_get_wtime();
    cout << "fieldSum time: " << t3 - t2 << endl;

    // ------------------------
    double atomicFlux = 0.0;
#pragma omp parallel for reduction(+:atomicFlux) schedule(static)
    for (int i = 0; i < numParticles; ++i)
        atomicFlux += velocity[i] * 1e-6;

    // criticalFlux：complex logic
    double criticalFlux = 0.0;
    for(int i = 0; i < numParticles; i++){
        // cal temp val
        double tempVal = sqrt( fabs(energy[i]) ) / 100.0;
        double extraVal = log(1 + fabs(velocity[i])) * 0.01;


        double oldValue = criticalFlux;
        // 如果 oldValue < 500，就用第一種方式計算 否則第二種
        if (oldValue < 500.0) {
            criticalFlux = oldValue + tempVal + extraVal;
        }
        else {
            criticalFlux = oldValue + sqrt(tempVal) - extraVal;
        }

    }
    double end_time = omp_get_wtime();


    // 6) print result to check correctness
    // ------------------------
    double sumVelocity = 0.0;
    double sumPressure = 0.0;
    double sumEnergy   = 0.0;

#pragma omp parallel for reduction(+:sumVelocity, sumPressure, sumEnergy) schedule(static)
    for (int i = 0; i < numParticles; ++i) {
        sumVelocity += velocity[i];
        sumPressure += pressure[i];
        sumEnergy   += energy[i];
    }

    // double end_time = omp_get_wtime();
    cout << "Computation Time: " << end_time - start_time << " seconds" << endl;


    // ------------------------
    cout << "=== result ===" << endl;
    // energy
    cout << "fieldValue      = " << fieldSum << endl;
    cout << "energy[0]       = " << energy[0] << endl;
    // sum
    cout << "Sum(velocity)   = " << sumVelocity << endl;
    cout << "Sum(pressure)   = " << sumPressure << endl;
    cout << "Sum(energy)     = " << sumEnergy << endl;
    // flux
    cout << "atomicFlux      = " << atomicFlux << endl;
    cout << "criticalFlux    = " << criticalFlux << endl;

    return 0;
}
