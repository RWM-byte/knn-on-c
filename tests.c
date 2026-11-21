#include "lab5.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Это только примеры. Закрытые тесты проверяют гораздо больше — дописывайте свои.

// Создание и удаление коллекции своего варианта — чтобы примеры ниже
// были одинаковыми для Vector и List.
#if ISU_ID % 2 == 0
static Collection *newCollection(void) {
    return createVector(0);
}

static int append(Collection *collection, void *item) {
    return appendVectorItem(collection, item);
}

static void *get(const Collection *collection, size_t index) {
    return getVectorItem(collection, index);
}

static void destroy(Collection *collection, FreeFn freeItem) {
    freeVector(collection, freeItem);
}
#else
static Collection *newCollection(void) {
    return createList();
}

static int append(Collection *collection, void *item) {
    return appendListItem(collection, item);
}

static void *get(const Collection *collection, size_t index) {
    return getListItem(collection, index);
}

static void destroy(Collection *collection, FreeFn freeItem) {
    freeList(collection, freeItem);
}
#endif

// Свой тип, о котором ни коллекция, ни kNN ничего не знают.
struct Point {
    double x;
    double y;
    int label;
};

static double pointDistance(const void *a, const void *b) {
    const struct Point *left = a;
    const struct Point *right = b;
    double dx = left->x - right->x;
    double dy = left->y - right->y;
    return dx * dx + dy * dy;
}

static int pointClass(const void *item) {
    return ((const struct Point *)item)->label;
}

void testCollectionExample(void) {
    struct Point a = {1, 2, 0};
    struct Point b = {3, 4, 1};
    Collection *collection = newCollection();
    assert(collection != NULL);
    assert(append(collection, &a) == 0);
    assert(append(collection, &b) == 0);
    assert(append(collection, NULL) == -1);
    assert(collection->size == 2);
    assert(get(collection, 1) == &b);  // тот же указатель, никаких копий
    assert(get(collection, 2) == NULL);

#if ISU_ID % 2 == 0
    assert(removeVectorItem(collection, 0) == &a);
#else
    assert(removeListItem(collection, 0) == &a);
#endif
    assert(collection->size == 1);
    destroy(collection, NULL);  // a и b на стеке — освобождать их нельзя

    // Объекты в куче: коллекция освободит их сама, если передать free
    collection = newCollection();
    for (int i = 0; i < 3; i++) {
        struct Point *heap = malloc(sizeof(struct Point));
        heap->x = i;
        heap->y = i;
        heap->label = 0;
        append(collection, heap);
    }
    destroy(collection, free);
}

void testKnnExample(void) {
    // Два «нуля» слева и три «единицы» справа
    struct Point points[] = {{0, 0, 0}, {1, 0, 0}, {9, 0, 1}, {10, 0, 1}, {11, 0, 1}};
    Collection *train = newCollection();
    for (int i = 0; i < 5; i++) {
        append(train, &points[i]);
    }
    struct Point query = {0.5, 0, -1};
    assert(predictKnn(train, &query, 1, pointDistance, pointClass) == 0);
    assert(predictKnn(train, &query, 3, pointDistance, pointClass) == 0);  // 0, 0, 1
    assert(predictKnn(train, &query, 5, pointDistance, pointClass) == 1);  // голосуют все
    assert(predictKnn(train, &query, 0, pointDistance, pointClass) == -1);
    assert(predictKnn(NULL, &query, 1, pointDistance, pointClass) == -1);
    destroy(train, NULL);
}

void testSplitExample(void) {
    struct Point points[8];
    Collection *data = newCollection();
    for (int i = 0; i < 8; i++) {
        points[i].x = i;
        points[i].y = 0;
        points[i].label = i < 4;  // четыре единицы и четыре нуля
        append(data, &points[i]);
    }
    Collection *train = NULL;
    Collection *test = NULL;
    assert(splitCollection(data, 0.75, 1, pointClass, &train, &test) == 0);
    assert(train->size == 6);  // по три объекта каждого класса
    assert(test->size == 2);
    int positive = 0;
    for (size_t i = 0; i < train->size; i++) {
        positive += pointClass(get(train, i));
    }
    assert(positive == 3);
    // train и test — те же объекты, что в data: освобождаем только сами коллекции
    destroy(train, NULL);
    destroy(test, NULL);
    destroy(data, NULL);
}

void testMetricsExample(void) {
    int actual[] = {1, 1, 0, 0};
    int predicted[] = {1, 0, 0, 1};
    struct Metrics metrics = computeMetrics(actual, predicted, 4);
    assert(fabs(metrics.accuracy - 0.5) < 1e-9);
    assert(fabs(metrics.precision - 0.5) < 1e-9);
    assert(fabs(metrics.recall - 0.5) < 1e-9);
    assert(fabs(metrics.f1 - 0.5) < 1e-9);

    double values[] = {1, 2, 3};
    double mean, std;
    assert(computeMeanStd(values, 3, &mean, &std) == 0);
    assert(fabs(mean - 2) < 1e-9);
    assert(fabs(std - 1) < 1e-9);  // делим на n - 1
}

void testPokemonExample(void) {
    Collection *pokemon = loadPokemon("data/pokemon.csv");
    assert(pokemon != NULL);
    printf("Покемонов загружено: %zu\n", pokemon->size);
    assert(pokemon->size == 800);
    const void *first = get(pokemon, 0);
    assert(pokemonClass(first) == 0);  // Bulbasaur — не легендарный
    assert(preparePokemon(pokemon, pokemon) == 0);
    assert(pokemonDistance(first, first) == 0);
    destroy(pokemon, freePokemon);
}

int main(void) {
    testCollectionExample();
    testKnnExample();
    testSplitExample();
    testMetricsExample();
    testPokemonExample();
    puts("Открытые тесты lab5 пройдены");
    return 0;
}
