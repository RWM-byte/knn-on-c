#ifndef LAB5_H
#define LAB5_H

#include "variant.h"
#include <stddef.h>

/* ========================= Указатели на функции ========================== */

// Освобождает один элемент коллекции.
typedef void (*FreeFn)(void *item);

// Расстояние между двумя объектами одного типа: >= 0, d(a, a) == 0, d(a, b) == d(b, a).
typedef double (*DistanceFn)(const void *a, const void *b);

// Класс объекта: неотрицательное число (в датасетах lab5 — 0 или 1).
typedef int (*ClassFn)(const void *item);

/* ============================ Чётный ISU ID ============================== */

// Динамический массив указателей. Сами объекты в нём не лежат —
// только их адреса, объекты живут там, где их выделили.
struct Vector {
    void **items;     // непрерывный блок под capacity указателей
    size_t size;      // сколько указателей лежит сейчас
    size_t capacity;  // под сколько указателей выделена память
};

struct Vector *createVector(size_t capacity);
int appendVectorItem(struct Vector *vector, void *item);
void *getVectorItem(const struct Vector *vector, size_t index);
void *removeVectorItem(struct Vector *vector, size_t index);
void freeVector(struct Vector *vector, FreeFn freeItem);

/* =========================== Нечётный ISU ID ============================= */

struct ListNode {
    void *item;
    struct ListNode *next;
};

// Односвязный список указателей с хвостом: append за O(1).
struct List {
    struct ListNode *head;  // первый узел, NULL — список пуст
    struct ListNode *tail;  // последний узел, NULL — список пуст
    size_t size;            // сколько узлов в списке
};

struct List *createList(void);
int appendListItem(struct List *list, void *item);
void *getListItem(const struct List *list, size_t index);
void *removeListItem(struct List *list, size_t index);
void freeList(struct List *list, FreeFn freeItem);

/* ============================== Коллекция ================================ */

// Коллекция вашего варианта. Всё, что ниже, пишется через Collection.
#if ISU_ID % 2 == 0
typedef struct Vector Collection;
#else
typedef struct List Collection;
#endif

/* ================================ kNN ==================================== */

// Класс большинства среди k ближайших к query объектов train.
int predictKnn(const Collection *train, const void *query, size_t k,
               DistanceFn distance, ClassFn getClass);

/* ======================= Разбиение и метрики ============================= */

// Стратифицированное разбиение data на train (доля trainShare) и test.
// В *train и *test — новые коллекции с теми же указателями, что в data.
int splitCollection(const Collection *data, double trainShare, unsigned long long seed,
                    ClassFn getClass, Collection **train, Collection **test);

// Метрики бинарной классификации, положительный класс — 1.
struct Metrics {
    double accuracy;
    double precision;
    double recall;
    double f1;
};

struct Metrics computeMetrics(const int *actual, const int *predicted, size_t count);

// Среднее и выборочное (n - 1) стандартное отклонение.
int computeMeanStd(const double *values, size_t count, double *mean, double *std);

/* =============================== Датасеты ================================ */

// Для каждого датасета: загрузчик, подготовка признаков по train,
// расстояние, класс и освобождение одного объекта. Структуры объектов —
// ваши, в solutions.c.

// data/pokemon.csv: легендарный ли покемон (Legendary == True -> 1).
Collection *loadPokemon(const char *path);
int preparePokemon(const Collection *train, const Collection *test);
double pokemonDistance(const void *a, const void *b);
int pokemonClass(const void *item);
void freePokemon(void *item);

// data/speed_dating.csv: взаимная симпатия после свидания (match == 1 -> 1).
Collection *loadSpeedDating(const char *path);
int prepareSpeedDating(const Collection *train, const Collection *test);
double speedDatingDistance(const void *a, const void *b);
int speedDatingClass(const void *item);
void freeSpeedDating(void *item);

// data/student.csv: сдал ли студент математику (G3 >= 10 -> 1).
Collection *loadStudents(const char *path);
int prepareStudents(const Collection *train, const Collection *test);
double studentDistance(const void *a, const void *b);
int studentClass(const void *item);
void freeStudent(void *item);

#endif
