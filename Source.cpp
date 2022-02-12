#include <iostream>
#include <immintrin.h>
#include <ctime>
#include <ratio>
#include <chrono>

#define SIZE 800
#define SIZE_X 2
#define SIZE_Y 1

using namespace std;

void comparMatrix();
void multiplicationClassic();
void multiplicationVector();
void multiplicationSSE1();

float MatrixA[SIZE * SIZE_Y][SIZE * SIZE_X];
float MatrixB[SIZE * SIZE_X][SIZE * SIZE_Y];
float MatrixC1[SIZE * SIZE_Y][SIZE * SIZE_Y];
float MatrixC2[SIZE * SIZE_Y][SIZE * SIZE_Y];
float MatrixC3[SIZE * SIZE_Y][SIZE * SIZE_Y];

int main()
{
    auto start = chrono::high_resolution_clock::now();
    multiplicationClassic();
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration <double> time1 = stop - start;
    cout << "time Classic: " << time1.count() << " seconds\n";

    start = chrono::high_resolution_clock::now();
    multiplicationVector();
    stop = chrono::high_resolution_clock::now();
    chrono::duration <double> time2 = stop - start;
    cout << "time Vector: " << time2.count() << " seconds\n";

    start = chrono::high_resolution_clock::now();
    multiplicationSSE1();
    stop = chrono::high_resolution_clock::now();
    chrono::duration <double> time3 = stop - start;
    cout << "time SSE1: " << time3.count() << " seconds\n";

    comparMatrix();

    return 0;
}

void multiplicationClassic()
{
#pragma loop(no_vector)
    for (int i = 0; i < SIZE * SIZE_Y; i++)
#pragma loop(no_vector)
        for (int k = 0; k < SIZE * SIZE_X; k++)
#pragma loop(no_vector)
            for (int j = 0; j < SIZE * SIZE_Y; j++)
                MatrixC2[i][j] += MatrixA[i][k] * MatrixB[k][j];
    return;
}

void multiplicationVector()
{
    for (int i = 0; i < SIZE * SIZE_Y; i++)
        for (int k = 0; k < SIZE * SIZE_X; k++)
            for (int j = 0; j < SIZE * SIZE_Y; j++)
                MatrixC1[i][j] += MatrixA[i][k] * MatrixB[k][j];
    return;
}

void multiplicationSSE1()
{
    __m128 sum = _mm_setzero_ps();
    for (int i = 0; i < SIZE * SIZE_Y; i++)
        for (int k = 0; k < SIZE * SIZE_X; k += 4)
        {
            for (int j = 0; j < SIZE * SIZE_Y; j++)
            {
                __m128 line = _mm_set1_ps(MatrixA[i][k]);     //считываем значение устанавливаем во все 4 позиции
                __m128 row = _mm_load_ps(&MatrixB[k][j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(line, row));
            }
            _mm_store_ps(&MatrixC3[i][k], sum);
        }
    return;
}

void comparMatrix()
{
    for (int i = 0; i < SIZE * SIZE_Y; ++i)
        for (int j = 0; j < SIZE * SIZE_Y; ++j)
            if (MatrixC1[i][j] != MatrixC2[i][j]) { std::cout << "MatrixC1 != MatrixC2"; return; }

    std::cout << endl << "MatrixC1 == MatrixC2";

    for (int i = 0; i < SIZE * SIZE_Y; ++i)
        for (int j = 0; j < SIZE * SIZE_Y; ++j)
            if (MatrixC1[i][j] != MatrixC3[i][j]) { std::cout << endl << "MatrixC1 != MatrixC3"; return; }

    std::cout << " == MatrixC3" << endl;

    return;
}