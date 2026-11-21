#include "knn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



#define RUNS 10


typedef Collection *(*LoadFn)(const char *);
typedef int (*PrepareFn)(const Collection *, const Collection *);
typedef double (*DistanceFn)(const void *, const void *);
typedef int (*ClassFnLocal)(const void *);


static int benchmarkDataset(
    const char *name,
    const char *path,
    LoadFn load,
    PrepareFn prepare,
    DistanceFn distance,
    ClassFnLocal getClass,
    FreeFn freeItem
) {
    static size_t ks[] = {1, 3, 5, 7, 9};

    clock_t startLoad = clock();
    Collection *data = load(path);
    clock_t endLoad = clock();

    double loadMs = (double)(endLoad - startLoad) * 1000.0 / CLOCKS_PER_SEC;

    if (data == NULL) return -1;
    if (data->size == 0) {
        freeVector(data, freeItem);
        return -1;
    }

    size_t positiveCount = 0;
    for (size_t i = 0; i < data->size; i++) {
        if (getClass(data->items[i]) == 1) {
            positiveCount++;
        }
    }

    printf("## %s: %zu объектов, класс 1 — %zu (%.1f%%), загрузка %.1f мс\n",
       name,
       data->size,
       positiveCount,
       100.0 * positiveCount / data->size,
       loadMs);
    
    printf("seed 1..10\n\n");

    for (size_t ki = 0; ki < 5; ki++) {
        size_t k = ks[ki];

        double accuracies[RUNS];
        double precisions[RUNS];
        double recalls[RUNS];
        double f1s[RUNS];
        double times[RUNS];
        double timesPerItem[RUNS];

        for (size_t seed = 1; seed <= RUNS; seed++) {
            Collection *train = NULL;
            Collection *test = NULL;

            if (splitCollection(
                    data, 0.75, seed, getClass,
                    &train, &test) != 0) {
                freeVector(data, freeItem);
                return -1;
            }

            if (prepare(train, test) != 0) {
                freeVector(train, NULL);
                freeVector(test, NULL);
                freeVector(data, freeItem);
                return -1;
            }

            if (ki == 0 && seed == 1) {
                printf("train %zu, test %zu\n\n", train->size, test->size);
                printf("| k | accuracy | precision | recall | F1 | test (ms) | per item (us) |\n");
            }

            size_t n = test->size;

            if (n == 0) {
                freeVector(train, NULL);
                freeVector(test, NULL);
                freeVector(data, freeItem);
                return -1;
            }

            int *predicted = malloc(n * sizeof(*predicted));
            int *actual = malloc(n * sizeof(*actual));

            if (predicted == NULL || actual == NULL) {
                free(predicted);
                free(actual);
                freeVector(train, NULL);
                freeVector(test, NULL);
                freeVector(data, freeItem);
                return -1;
            }

            clock_t start = clock();

            for (size_t i = 0; i < n; i++) {
                predicted[i] = predictKnn(
                    train,
                    test->items[i],
                    k,
                    distance,
                    getClass
                );
            }

            clock_t end = clock();

            for (size_t i = 0; i < n; i++) {
                if (predicted[i] < 0) {
                    free(predicted);
                    free(actual);
                    freeVector(train, NULL);
                    freeVector(test, NULL);
                    freeVector(data, freeItem);
                    return -1;
                }
            }

            double elapsedMs = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

            for (size_t i = 0; i < n; i++) {
                actual[i] = getClass(test->items[i]);
            }

            struct Metrics metrics =
                computeMetrics(actual, predicted, n);

            size_t index = seed - 1;

            accuracies[index] = metrics.accuracy;
            precisions[index] = metrics.precision;
            recalls[index] = metrics.recall;
            f1s[index] = metrics.f1;

            times[index] = elapsedMs;
            timesPerItem[index] = elapsedMs * 1000.0 / n;

            free(predicted);
            free(actual);

            freeVector(train, NULL);
            freeVector(test, NULL);
        }

        double meanAcc, stdAcc;
        double meanPrec, stdPrec;
        double meanRec, stdRec;
        double meanF1, stdF1;
        double meanTime, stdTime;
        double meanPerItem, stdPerItem;

        computeMeanStd(accuracies, RUNS, &meanAcc, &stdAcc);
        computeMeanStd(precisions, RUNS, &meanPrec, &stdPrec);
        computeMeanStd(recalls, RUNS, &meanRec, &stdRec);
        computeMeanStd(f1s, RUNS, &meanF1, &stdF1);
        computeMeanStd(times, RUNS, &meanTime, &stdTime);
        computeMeanStd(timesPerItem, RUNS,
                       &meanPerItem, &stdPerItem);

        printf("| %zu | %.3f +/- %.3f | %.3f +/- %.3f | "
               "%.3f +/- %.3f | %.3f +/- %.3f | "
               "%.3f +/- %.3f | %.3f +/- %.3f |\n",
               k,
               meanAcc, stdAcc,
               meanPrec, stdPrec,
               meanRec, stdRec,
               meanF1, stdF1,
               meanTime, stdTime,
               meanPerItem, stdPerItem);
    }

    freeVector(data, freeItem);
    return 0;
}

int main(void) {
    if (benchmarkDataset(
            "Pokemon",
            "data/pokemon.csv",
            loadPokemon,
            preparePokemon,
            pokemonDistance,
            pokemonClass,
            freePokemon
        ) != 0) {
        return 1;
    }

    if (benchmarkDataset(
            "Speed Dating",
            "data/speed_dating.csv",
            loadSpeedDating,
            prepareSpeedDating,
            speedDatingDistance,
            speedDatingClass,
            freeSpeedDating
        ) != 0) {
        return 1;
    }

    if (benchmarkDataset(
            "Student Performance",
            "data/student.csv",
            loadStudents,
            prepareStudents,
            studentDistance,
            studentClass,
            freeStudent
        ) != 0) {
        return 1;
    }

    return 0;
}
