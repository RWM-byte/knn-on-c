#include "knn.h"
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define FEATURES_POKEMON 8
#define FEATURES_DATES 58
#define FEATURES_STUDENTS 13


static unsigned long long nextRandom(unsigned long long *state) {
    unsigned long long value = (*state += 0x9E3779B97F4A7C15ULL);

    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;

    return value ^ (value >> 31);
}

static void fisherJets(void* arr[], size_t n, unsigned long long seed) {
    if (n < 2) return;

    for (size_t i = n - 1; i > 0; i--) {
        size_t j = nextRandom(&seed) % (i + 1);
        void* tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

struct Vector *createVector(size_t capacity) {
    if (capacity > SIZE_MAX / sizeof(void *)) return NULL;

    struct Vector *vector = (struct Vector *)malloc(sizeof(*vector));
    if (vector == NULL) return NULL;

    vector->capacity = capacity;
    vector->size = 0;
    vector->items = NULL;

    if (capacity > 0) {
        vector->items = malloc(sizeof(void*) * capacity);
        if (vector->items == NULL) {
            free(vector);
            return NULL;
        }
    }

    return vector;
}


int appendVectorItem(struct Vector *vector, void *item) {
    if (vector == NULL || item == NULL) return -1;

    if (vector->size == vector->capacity) {
        size_t newCapacity;
        if (vector->capacity == 0) newCapacity = 1;
        else {
            if (vector->capacity > SIZE_MAX / 2) return -1;
            newCapacity = vector->capacity * 2;
        }
        if (newCapacity > SIZE_MAX / sizeof(void *)) return -1;

        void **temp = realloc(vector->items, newCapacity * sizeof(void *));
        if (temp == NULL) return -1;
        vector->items = temp;
        vector->capacity = newCapacity;
    }

    vector->items[vector->size] = item;
    vector->size++;

    return 0;
}

void *getVectorItem(const struct Vector *vector, size_t index) {
    if (vector == NULL || vector->size <= index) return NULL;
    return vector->items[index];
}


void *removeVectorItem(struct Vector *vector, size_t index) {
    if (vector == NULL || vector->size <= index) return NULL;
    void *tmp = vector->items[index];
    for (size_t i = index; i < vector->size - 1; i++) {
        vector->items[i] = vector->items[i+1];
    }
    vector->size--;
    return tmp;
}


void freeVector(struct Vector *vector, FreeFn freeItem) {
    if (vector == NULL) return;
    if(freeItem != NULL) {
        for (size_t i = 0; i < vector->size; i++) {
            freeItem(vector->items[i]);
        }
    }
    free(vector->items);
    free(vector);
}


typedef struct Pokemon {
    double raw[8];
    double features[8];  
    unsigned int legendary:1;
} Pokemon;


static int parseCsv(char *line, char **fields, size_t capacity,
                    char delimiter, int singleQuotes) {
    char *read = line;
    char *write = line;
    size_t count = 0;

    while (1) {
        if (count == capacity) return -1;
        fields[count++] = write;

        if (*read == '"' || (singleQuotes && *read == '\'')) {
            char quote = *read++;
            while (1) {
                if (*read == '\0') return -1;
                if (*read == quote) {
                    if (read[1] == quote) {
                        *write++ = quote;
                        read += 2;
                    } else {
                        read++;
                        break;
                    }
                } else {
                    *write++ = *read++;
                }
            }
            if (*read != delimiter && *read != '\0') return -1;
        } else {
            while (*read != delimiter && *read != '\0') {
                if (*read == '"') return -1;
                *write++ = *read++;
            }
        }

        char separator = *read;
        if (separator == delimiter) read++;
        *write++ = '\0';
        if (separator == '\0') break;
    }

    return (int)count;
}

static int readCsvRow(FILE *file, char *line, size_t capacity,
                      char **fields, size_t maxFields,
                      char delimiter, int singleQuotes) {
    while (fgets(line, (int)capacity, file) != NULL) {
        size_t length = strlen(line);
        if (length == capacity - 1 && line[length - 1] != '\n') return -1;
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;
        return parseCsv(line, fields, maxFields, delimiter, singleQuotes);
    }
    return ferror(file) ? -1 : 0;
}

static int findColumns(char *fields[], int fieldCount,
                       const char *names[], size_t count, size_t columns[]) {
    for (size_t i = 0; i < count; i++) {
        int j = 0;
        while (j < fieldCount && strcmp(fields[j], names[i]) != 0) j++;
        if (j == fieldCount) return -1;
        columns[i] = (size_t)j;
    }
    return 0;
}

static double readNumber(const char *text) {
    char *end;
    errno = 0;
    double value = strtod(text, &end);
    if (end == text || errno == ERANGE || !isfinite(value)) return NAN;
    while (isspace((unsigned char)*end)) end++;
    if (*end != '\0') return NAN;
    return value;
}

static int readFeatures(char *fields[], int fieldCount,
                        const size_t columns[], size_t count, double raw[]) {
    for (size_t i = 0; i < count; i++) {
        if (columns[i] >= (size_t)fieldCount) return -1;
        raw[i] = readNumber(fields[columns[i]]);
        if (!isfinite(raw[i])) return -1;
    }
    return 0;
}

Collection *loadPokemon(const char *path) {
    if (path == NULL) return NULL;
    FILE *file = fopen(path, "r");
    if (file == NULL) return NULL;

    const char *names[] = {
        "Total", "HP", "Attack", "Defense", "Sp. Atk",
        "Sp. Def", "Speed", "Generation", "Legendary"
    };
    size_t columns[9];
    char line[4096];
    char *fields[256];
    int fieldCount = readCsvRow(file, line, sizeof(line), fields, 256, ',', 0);
    if (fieldCount <= 0 || findColumns(fields, fieldCount, names, 9, columns) != 0) {
        fclose(file);
        return NULL;
    }

    Collection *data = createVector(0);
    if (data == NULL) {
        fclose(file);
        return NULL;
    }

    while ((fieldCount = readCsvRow(file, line, sizeof(line), fields, 256, ',', 0)) > 0) {
        Pokemon object;
        if (readFeatures(fields, fieldCount, columns, 8, object.raw) != 0) continue;
        if (columns[8] >= (size_t)fieldCount) continue;
        if (strcmp(fields[columns[8]], "True") == 0) object.legendary = 1;
        else if (strcmp(fields[columns[8]], "False") == 0) object.legendary = 0;
        else continue;
        memcpy(object.features, object.raw, sizeof(object.features));

        Pokemon *item = malloc(sizeof(*item));
        if (item == NULL) {
            fclose(file);
            freeVector(data, free);
            return NULL;
        }
        *item = object;
        if (appendVectorItem(data, item) != 0) {
            free(item);
            fclose(file);
            freeVector(data, free);
            return NULL;
        }
    }

    fclose(file);
    if (fieldCount < 0) {
        freeVector(data, free);
        return NULL;
    }
    return data;
}

int preparePokemon(const Collection *train, const Collection *test) {
    if (train == NULL || test == NULL) {
        return -1;
    }

    double mean[FEATURES_POKEMON];
    double std[FEATURES_POKEMON];
    for (size_t i = 0; i < FEATURES_POKEMON; i++) {
        double sum = 0.0;
        size_t count = 0;

       for (size_t j = 0; j < train->size; j++) {
            const Pokemon *item = train->items[j];

            if (isfinite(item->raw[i])) {
                sum += item->raw[i];
                count++;
            }
        }

        mean[i] = count > 0 ? sum / count : 0.0;

        double sumSquares = 0.0;
        size_t count1 = 0;

       for (size_t j = 0; j < train->size; j++) {
            const Pokemon *item = train->items[j];

            if (isfinite(item->raw[i])) {
                double diff = item->raw[i] - mean[i];
                sumSquares += diff * diff;
                count1++;
            }
        }

        std[i] = count1 > 0 ? sqrt(sumSquares / count1) : 0.0;
    }

    for (size_t j = 0; j < train->size; j++) {
        Pokemon *item = train->items[j];

        for (size_t i = 0; i < FEATURES_POKEMON; i++) {
            double value = item->raw[i];

            if (!isfinite(value)) {
                value = mean[i];
            }

            item->features[i] = std[i] > 0.0
                ? (value - mean[i]) / std[i]
                : 0.0;
        }
    }

    for (size_t j = 0; j < test->size; j++) {
        Pokemon *item = test->items[j];

        for (size_t i = 0; i < FEATURES_POKEMON; i++) {
            double value = item->raw[i];

            if (!isfinite(value)) {
                value = mean[i];
            }

            item->features[i] = std[i] > 0.0
                ? (value - mean[i]) / std[i]
                : 0.0;
        }
    }

    return 0;
}

double pokemonDistance(const void *a, const void *b) {
    const Pokemon* first = (Pokemon*)a;
    const Pokemon* second = (Pokemon*)b;

    double dist = 0.0;

    for (size_t i = 0; i < FEATURES_POKEMON; i++) {
        dist += pow((second->features[i] - first->features[i]),2);
    }

    return dist;
}

int pokemonClass(const void *item) {
    const Pokemon* tmp = (Pokemon*)item;
    return tmp->legendary;
}

void freePokemon(void *item) {
    free(item);
}


typedef struct Dating {
    double raw[58];
    double features[58];
    unsigned int soulmate:1;
} Dating;

Collection *loadSpeedDating(const char *path) {
    if (path == NULL) return NULL;
    FILE *file = fopen(path, "r");
    if (file == NULL) return NULL;

    const char *names[] = {
        "age", "age_o", "samerace",
        "importance_same_race", "importance_same_religion", "pref_o_attractive",
        "pref_o_sincere", "pref_o_intelligence", "pref_o_funny",
        "pref_o_ambitious", "pref_o_shared_interests", "attractive_o",
        "sinsere_o", "intelligence_o", "funny_o",
        "ambitous_o", "shared_interests_o", "attractive_important",
        "sincere_important", "intellicence_important", "funny_important",
        "ambtition_important", "shared_interests_important", "attractive",
        "sincere", "intelligence", "funny",
        "ambition", "attractive_partner", "sincere_partner",
        "intelligence_partner", "funny_partner", "ambition_partner",
        "shared_interests_partner", "sports", "tvsports",
        "exercise", "dining", "museums",
        "art", "hiking", "gaming",
        "clubbing", "reading", "tv",
        "theater", "movies", "concerts",
        "music", "shopping", "yoga",
        "interests_correlate", "expected_happy_with_sd_people", "expected_num_interested_in_me",
        "expected_num_matches", "like", "guess_prob_liked",
        "met", "match"
    };
    size_t columns[59];
    char line[4096];
    char *fields[256];
    int fieldCount = readCsvRow(file, line, sizeof(line), fields, 256, ',', 1);
    if (fieldCount <= 0 || findColumns(fields, fieldCount, names, 59, columns) != 0) {
        fclose(file);
        return NULL;
    }

    Collection *data = createVector(0);
    if (data == NULL) {
        fclose(file);
        return NULL;
    }

    while ((fieldCount = readCsvRow(file, line, sizeof(line), fields, 256, ',', 1)) > 0) {
        Dating object;
        if (readFeatures(fields, fieldCount, columns, 58, object.raw) != 0) continue;
        if (columns[58] >= (size_t)fieldCount) continue;
        double target = readNumber(fields[columns[58]]);
        if (target != 0.0 && target != 1.0) continue;
        object.soulmate = target == 1.0;
        memcpy(object.features, object.raw, sizeof(object.features));

        Dating *item = malloc(sizeof(*item));
        if (item == NULL) {
            fclose(file);
            freeVector(data, free);
            return NULL;
        }
        *item = object;
        if (appendVectorItem(data, item) != 0) {
            free(item);
            fclose(file);
            freeVector(data, free);
            return NULL;
        }
    }

    fclose(file);
    if (fieldCount < 0) {
        freeVector(data, free);
        return NULL;
    }
    return data;
}

int prepareSpeedDating(const Collection *train, const Collection *test) {
    if (train == NULL || test == NULL) {
        return -1;
    }

    double mean[FEATURES_DATES];
    double std[FEATURES_DATES];
    for (size_t i = 0; i < FEATURES_DATES; i++) {
        double sum = 0.0;
        size_t count = 0;

       for (size_t j = 0; j < train->size; j++) {
            const Dating *item = train->items[j];

            if (isfinite(item->raw[i])) {
                sum += item->raw[i];
                count++;
            }
        }

        mean[i] = count > 0 ? sum / count : 0.0;

        double sumSquares = 0.0;
        size_t count1 = 0;

       for (size_t j = 0; j < train->size; j++) {
            const Dating *item = train->items[j];

            if (isfinite(item->raw[i])) {
                double diff = item->raw[i] - mean[i];
                sumSquares += diff * diff;
                count1++;
            }
        }

        std[i] = count1 > 0 ? sqrt(sumSquares / count1) : 0.0;
    }

    for (size_t j = 0; j < train->size; j++) {
        Dating *item = train->items[j];

        for (size_t i = 0; i < FEATURES_DATES; i++) {
            double value = item->raw[i];

            if (!isfinite(value)) {
                value = mean[i];
            }

            item->features[i] = std[i] > 0.0
                ? (value - mean[i]) / std[i]
                : 0.0;
        }
    }

    for (size_t j = 0; j < test->size; j++) {
        Dating *item = test->items[j];

        for (size_t i = 0; i < FEATURES_DATES; i++) {
            double value = item->raw[i];

            if (!isfinite(value)) {
                value = mean[i];
            }

            item->features[i] = std[i] > 0.0
                ? (value - mean[i]) / std[i]
                : 0.0;
        }
    }

    return 0;
}
double speedDatingDistance(const void *a, const void *b) {
    const Dating* first = (Dating*)a;
    const Dating* second = (Dating*)b;

    double dist = 0.0;

    for (size_t i = 0; i < FEATURES_DATES; i++) {
        dist += pow((second->features[i]- first->features[i]),2);
    }

    return dist;
}
int speedDatingClass(const void *item) {
    const Dating* tmp = (Dating*)item;
    return tmp->soulmate;
}
void freeSpeedDating(void *item) {
     free(item);
}


typedef struct Student {
    double raw[13];
    double features[13];
    unsigned int grade:1;
} Student;


Collection *loadStudents(const char *path) {
    if (path == NULL) return NULL;
    FILE *file = fopen(path, "r");
    if (file == NULL) return NULL;

    const char *names[] = {
        "age", "Medu", "Fedu",
        "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout",
        "Dalc", "Walc", "health",
        "absences", "G3"
    };
    size_t columns[14];
    char line[4096];
    char *fields[256];
    int fieldCount = readCsvRow(file, line, sizeof(line), fields, 256, ';', 0);
    if (fieldCount <= 0 || findColumns(fields, fieldCount, names, 14, columns) != 0) {
        fclose(file);
        return NULL;
    }

    Collection *data = createVector(0);
    if (data == NULL) {
        fclose(file);
        return NULL;
    }

    while ((fieldCount = readCsvRow(file, line, sizeof(line), fields, 256, ';', 0)) > 0) {
        Student object;
        if (readFeatures(fields, fieldCount, columns, 13, object.raw) != 0) continue;
        if (columns[13] >= (size_t)fieldCount) continue;
        double target = readNumber(fields[columns[13]]);
        if (!isfinite(target) || target < 0 || target > 20 || floor(target) != target) continue;
        object.grade = target >= 10;
        memcpy(object.features, object.raw, sizeof(object.features));

        Student *item = malloc(sizeof(*item));
        if (item == NULL) {
            fclose(file);
            freeVector(data, free);
            return NULL;
        }
        *item = object;
        if (appendVectorItem(data, item) != 0) {
            free(item);
            fclose(file);
            freeVector(data, free);
            return NULL;
        }
    }

    fclose(file);
    if (fieldCount < 0) {
        freeVector(data, free);
        return NULL;
    }
    return data;
}

int prepareStudents(const Collection *train, const Collection *test) {
    if (train == NULL || test == NULL) {
        return -1;
    }

    double mean[FEATURES_STUDENTS];
    double std[FEATURES_STUDENTS];
    for (size_t i = 0; i < FEATURES_STUDENTS; i++) {
        double sum = 0.0;
        size_t count = 0;

       for (size_t j = 0; j < train->size; j++) {
            const Student *item = train->items[j];

            if (isfinite(item->raw[i])) {
                sum += item->raw[i];
                count++;
            }
        }

        mean[i] = count > 0 ? sum / count : 0.0;

        double sumSquares = 0.0;
        size_t count1 = 0;

       for (size_t j = 0; j < train->size; j++) {
            const Student *item = train->items[j];

            if (isfinite(item->raw[i])) {
                double diff = item->raw[i] - mean[i];
                sumSquares += diff * diff;
                count1++;
            }
        }

        std[i] = count1 > 0 ? sqrt(sumSquares / count1) : 0.0;
    }

    for (size_t j = 0; j < train->size; j++) {
        Student *item = train->items[j];

        for (size_t i = 0; i < FEATURES_STUDENTS; i++) {
            double value = item->raw[i];

            if (!isfinite(value)) {
                value = mean[i];
            }

            item->features[i] = std[i] > 0.0
                ? (value - mean[i]) / std[i]
                : 0.0;
        }
    }

    for (size_t j = 0; j < test->size; j++) {
        Student *item = test->items[j];

        for (size_t i = 0; i < FEATURES_STUDENTS; i++) {
            double value = item->raw[i];

            if (!isfinite(value)) {
                value = mean[i];
            }

            item->features[i] = std[i] > 0.0
                ? (value - mean[i]) / std[i]
                : 0.0;
        }
    }

    return 0;
}

double studentDistance(const void *a, const void *b) {
    const Student* first = (Student*)a;
    const Student* second = (Student*)b;

    double dist = 0.0;

    for (size_t i = 0; i < FEATURES_STUDENTS; i++) {
        dist += pow((second->features[i]- first->features[i]),2);
    }

    return dist;
}
int studentClass(const void *item) {
    const Student* tmp = (Student*)item;
    return tmp->grade;
}
void freeStudent(void *item) {
     free(item);
}




int predictKnn(const Collection *train, const void *query, size_t k,
               DistanceFn distance, ClassFn getClass) 
{
    if (train == NULL || query == NULL || distance == NULL || 
        getClass == NULL || k == 0 || train->size == 0) return -1;
    
    k = (k > train->size) ? (train->size) : (k);

    if (k > SIZE_MAX / sizeof(double) || k > SIZE_MAX / sizeof(void *)) return -1;

    double *bestDistance = malloc(k * sizeof(*bestDistance));
    void **bestItems = malloc(k * sizeof(*bestItems));
    if (bestDistance == NULL || bestItems == NULL) {
        free(bestDistance);
        free(bestItems);
        return -1;
    }
    
    for (size_t i = 0; i < k; i++) {
        bestDistance[i] = INFINITY;
        bestItems[i] = NULL;
    }

    for (size_t i = 0; i < train->size; i++) {
        double dist = distance(query, train->items[i]);

        if (dist > bestDistance[k-1]) continue;

        size_t j = k - 1;
        while (j > 0 && dist < bestDistance[j - 1]) {
            bestDistance[j] = bestDistance[j - 1];
            bestItems[j] = bestItems[j - 1];
            j--;
        }

        bestDistance[j] = dist;
        bestItems[j] = train->items[i];
    }

    int bestClass = -1;
    size_t max_votes = 0;

    for (size_t i = 0; i < k; i++) {
        int currentClass = getClass(bestItems[i]);
        size_t votes = 0;

        for (size_t j = 0; j < k; j++) {
            if (getClass(bestItems[j]) == currentClass) {
                votes++;
            }
        }

        if (votes > max_votes) {
            max_votes = votes;
            bestClass = currentClass;
        }
    }

    free(bestDistance);
    free(bestItems);
    return bestClass;
}


int splitCollection(const Collection *data, double trainShare, unsigned long long seed,
                    ClassFn getClass, Collection **train, Collection **test)
{
    if (data == NULL || getClass == NULL || train == NULL || test == NULL ) return -1;
    if (!(trainShare > 0.0 && trainShare < 1.0)) return -1;

    if (train == test) return -1;
    if (data->size > SIZE_MAX / sizeof(void *)) return -1;

    void **tmp = NULL;
    if (data->size > 0) {
        tmp = malloc(data->size * sizeof(*tmp));
        if (tmp == NULL) return -1;
    }

    size_t positiveClass = 0, negativeClass = 0;
    for (size_t i = 0; i < data->size; i++) {
        tmp[i] = data->items[i];
        if (getClass(data->items[i]) == 1) positiveClass++;
        else negativeClass++;
    }
    size_t quotePositive = (size_t)round(positiveClass * trainShare);
    size_t quoteNegative = (size_t)round(negativeClass * trainShare);

    fisherJets(tmp, data->size, seed);

    Collection *newTrain = createVector(0);
    Collection *newTest = createVector(0);
    if (newTrain == NULL || newTest == NULL) {
        freeVector(newTrain, NULL);
        freeVector(newTest, NULL);
        free(tmp);
        return -1;
    }

    size_t k = 0, m = 0;
    for (size_t i = 0; i < data->size; i++) {
        Collection *target = newTest;
        if (getClass(tmp[i]) == 1) {
            if (k < quotePositive) {
                target = newTrain;
                k++;
            }
        } else {
            if (m < quoteNegative) {
                target = newTrain;
                m++;
            }
        }
        if (appendVectorItem(target, tmp[i]) != 0) {
            freeVector(newTrain, NULL);
            freeVector(newTest, NULL);
            free(tmp);
            return -1;
        }
    }

    free(tmp);
    *train = newTrain;
    *test = newTest;

    return 0;
}



struct Metrics computeMetrics(const int *actual,
                              const int *predicted,
                              size_t count) {

    struct Metrics metrics;

    if (actual == NULL || predicted == NULL || count == 0) {
        metrics.accuracy = NAN;
        metrics.precision = NAN;
        metrics.recall = NAN;
        metrics.f1 = NAN;
        return metrics;
    }

    size_t TP = 0, FP = 0, TN = 0, FN = 0;

    for (size_t i = 0; i < count; i++) {
        if (actual[i] == 1 && predicted[i] == 1) TP++;
        else if (actual[i] != 1 && predicted[i] != 1) TN++;
        else if (actual[i] == 1 && predicted[i] != 1) FN++;
        else  FP++;
    }

    double accuracy = (double)(TP + TN) / count;
    double precision = (TP + FP == 0) ? 0.0 : (double)TP / (TP + FP);
    double recall = (TP + FN == 0) ? 0.0 : (double)TP / (TP + FN);
    double F1 = (precision + recall == 0) ? 0.0 : 2.0 * precision * recall / (precision + recall);
    metrics.accuracy = accuracy;
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.f1 = F1;

    return metrics;
}

 
int computeMeanStd(const double *values, size_t count,
                   double *mean, double *std) {
    if (values == NULL || mean == NULL || std == NULL || count == 0) return -1;

    double sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    double avg = sum / count;

    double deviation = 0;
    if (count > 1) {
        double sumSquares = 0;
        for (size_t i = 0; i < count; i++) {
            double diff = values[i] - avg;
            sumSquares += diff * diff;
        }

        deviation = sqrt(sumSquares / (count - 1));
    }


    *mean = avg;
    *std = deviation;

    return 0;
}
