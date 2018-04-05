#define REAL float
#define INT int

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <pthread.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

const REAL pi = 3.141592653589793238462643383;
const INT SIGMOID_BOUND = 6;
const INT sigmoid_table_size = 1000;

INT threads = 16;
INT bernFlag = 0;
INT loadBinaryFlag = 0;
INT outBinaryFlag = 0;
INT trainTimes = 1000;
INT nbatches = 100;
INT dimension = 100;
REAL alpha = 0.001;
REAL margin = 0.5;

REAL con_lambda = 0.0625;

string inPath = "../data/FB15K/";
string outPath = "";
string loadPath = "";
string initPath = "";
string note = "";

INT *lefHead, *rigHead;
INT *lefTail, *rigTail;
REAL* sigmoid_table;

struct Triple {
    INT h, r, t;
};

Triple *trainHead, *trainTail, *trainList;

struct cmp_head {
    bool operator()(const Triple& a, const Triple& b)
    {
	return (a.h < b.h) || (a.h == b.h && a.r < b.r) || (a.h == b.h && a.r == b.r && a.t < b.t);
    }
};

struct cmp_tail {
    bool operator()(const Triple& a, const Triple& b)
    {
	return (a.t < b.t) || (a.t == b.t && a.r < b.r) || (a.t == b.t && a.r == b.r && a.h < b.h);
    }
};

/*
	There are some math functions for the program initialization.
*/
unsigned long long* next_random;

unsigned long long randd(INT id)
{
    next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
    return next_random[id];
}

INT rand_max(INT id, INT x)
{
    INT res = randd(id) % x;
    while (res < 0)
	res += x;
    return res;
}

REAL rand(REAL min, REAL max)
{
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

REAL normal(REAL x, REAL miu, REAL sigma)
{
    return 1.0 / sqrt(2 * pi) / sigma * exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma));
}

REAL randn(REAL miu, REAL sigma, REAL min, REAL max)
{
    REAL x, y, dScope;
    do {
	x = rand(min, max);
	y = normal(x, miu, sigma);
	dScope = rand(0.0, normal(miu, miu, sigma));
    } while (dScope > y);
    return x;
}

void norm(REAL* con)
{
    REAL x = 0;
    for (INT ii = 0; ii < dimension; ii++)
	x += (*(con + ii)) * (*(con + ii));
    x = sqrt(x);
    if (x > 1)
	for (INT ii = 0; ii < dimension; ii++)
	    *(con + ii) /= x;
}

void InitSigmoidTable()
{
    REAL x;
    sigmoid_table = (REAL*)malloc((sigmoid_table_size + 1) * sizeof(REAL));
    for (INT k = 0; k != sigmoid_table_size; k++) {
	x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
	sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

REAL FastSigmoid(REAL x)
{
    if (x > SIGMOID_BOUND)
	return 1;
    else if (x < -SIGMOID_BOUND)
	return 0;
    INT k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

/*
	Read triples from the training file.
*/

INT relationTotal, entityTotal, tripleTotal;
REAL *relationVec, *entityVec;
REAL *relationVecDao, *entityVecDao;
INT *freqRel, *freqEnt;
REAL *left_mean, *right_mean;

void init()
{

    FILE* fin;
    INT tmp;

    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%d", &relationTotal);
    fclose(fin);

    relationVec = (REAL*)calloc(relationTotal * dimension, sizeof(REAL));
    for (INT i = 0; i < relationTotal; i++) {
	for (INT ii = 0; ii < dimension; ii++)
	    relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
    }

    fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%d", &entityTotal);
    fclose(fin);

    entityVec = (REAL*)calloc(entityTotal * dimension, sizeof(REAL));
    for (INT i = 0; i < entityTotal; i++) {
	for (INT ii = 0; ii < dimension; ii++)
	    entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	norm(entityVec + i * dimension);
    }

    freqRel = (INT*)calloc(relationTotal + entityTotal, sizeof(INT));
    freqEnt = freqRel + relationTotal;

    fin = fopen((inPath + "train2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%d", &tripleTotal);
    trainHead = (Triple*)calloc(tripleTotal, sizeof(Triple));
    trainTail = (Triple*)calloc(tripleTotal, sizeof(Triple));
    trainList = (Triple*)calloc(tripleTotal, sizeof(Triple));
    for (INT i = 0; i < tripleTotal; i++) {
	tmp = fscanf(fin, "%d", &trainList[i].h);
	tmp = fscanf(fin, "%d", &trainList[i].t);
	tmp = fscanf(fin, "%d", &trainList[i].r);
	freqEnt[trainList[i].t]++;
	freqEnt[trainList[i].h]++;
	freqRel[trainList[i].r]++;
	trainHead[i] = trainList[i];
	trainTail[i] = trainList[i];
    }
    fclose(fin);

    sort(trainHead, trainHead + tripleTotal, cmp_head());
    sort(trainTail, trainTail + tripleTotal, cmp_tail());

    lefHead = (INT*)calloc(entityTotal, sizeof(INT));
    rigHead = (INT*)calloc(entityTotal, sizeof(INT));
    lefTail = (INT*)calloc(entityTotal, sizeof(INT));
    rigTail = (INT*)calloc(entityTotal, sizeof(INT));
    memset(rigHead, -1, sizeof(INT) * entityTotal);
    memset(rigTail, -1, sizeof(INT) * entityTotal);
    for (INT i = 1; i < tripleTotal; i++) {
	if (trainTail[i].t != trainTail[i - 1].t) {
	    rigTail[trainTail[i - 1].t] = i - 1;
	    lefTail[trainTail[i].t] = i;
	}
	if (trainHead[i].h != trainHead[i - 1].h) {
	    rigHead[trainHead[i - 1].h] = i - 1;
	    lefHead[trainHead[i].h] = i;
	}
    }
    rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
    rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

    left_mean = (REAL*)calloc(relationTotal * 2, sizeof(REAL));
    right_mean = left_mean + relationTotal;
    for (INT i = 0; i < entityTotal; i++) {
	for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
	    if (trainHead[j].r != trainHead[j - 1].r)
		left_mean[trainHead[j].r] += 1.0;
	if (lefHead[i] <= rigHead[i])
	    left_mean[trainHead[lefHead[i]].r] += 1.0;
	for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
	    if (trainTail[j].r != trainTail[j - 1].r)
		right_mean[trainTail[j].r] += 1.0;
	if (lefTail[i] <= rigTail[i])
	    right_mean[trainTail[lefTail[i]].r] += 1.0;
    }

    for (INT i = 0; i < relationTotal; i++) {
	left_mean[i] = freqRel[i] / left_mean[i];
	right_mean[i] = freqRel[i] / right_mean[i];
    }

    relationVecDao = (REAL*)calloc(dimension * relationTotal, sizeof(REAL));
    entityVecDao = (REAL*)calloc(dimension * entityTotal, sizeof(REAL));

    if (initPath != "") {
	FILE* f1 = fopen((initPath + "entity2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < entityTotal; i++) {
	    for (INT ii = 0; ii < dimension; ii++)
		tmp = fscanf(f1, "%f", &entityVec[i * dimension + ii]);
	    norm(entityVec + i * dimension);
	}
	fclose(f1);
	FILE* f2 = fopen((initPath + "relation2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < relationTotal; i++) {
	    for (INT ii = 0; ii < dimension; ii++)
		tmp = fscanf(f2, "%f", &relationVec[i * dimension + ii]);
	}
	fclose(f2);
    }

    InitSigmoidTable();
}

void load_binary()
{
    struct stat statbuf1;
    if (stat((loadPath + "entity2vec" + note + ".bin").c_str(), &statbuf1) != -1) {
	INT fd = open((loadPath + "entity2vec" + note + ".bin").c_str(), O_RDONLY);
	REAL* entityVecTmp = (REAL*)mmap(NULL, statbuf1.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	memcpy(entityVec, entityVecTmp, statbuf1.st_size);
	munmap(entityVecTmp, statbuf1.st_size);
	close(fd);
    }
    struct stat statbuf2;
    if (stat((loadPath + "relation2vec" + note + ".bin").c_str(), &statbuf2) != -1) {
	INT fd = open((loadPath + "relation2vec" + note + ".bin").c_str(), O_RDONLY);
	REAL* relationVecTmp = (REAL*)mmap(NULL, statbuf2.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	memcpy(relationVec, relationVecTmp, statbuf2.st_size);
	munmap(relationVecTmp, statbuf2.st_size);
	close(fd);
    }
}

void load()
{
    if (loadBinaryFlag) {
	load_binary();
	return;
    }
    FILE* fin;
    INT tmp;
    fin = fopen((loadPath + "entity2vec" + note + ".vec").c_str(), "r");
    for (INT i = 0; i < entityTotal; i++) {
	INT last = i * dimension;
	for (INT j = 0; j < dimension; j++)
	    tmp = fscanf(fin, "%f", &entityVec[last + j]);
    }
    fclose(fin);
    fin = fopen((loadPath + "relation2vec" + note + ".vec").c_str(), "r");
    for (INT i = 0; i < relationTotal; i++) {
	INT last = i * dimension;
	for (INT j = 0; j < dimension; j++)
	    tmp = fscanf(fin, "%f", &relationVec[last + j]);
    }
    fclose(fin);
}

/*
	Training process of transE.
*/

INT Len;
INT Batch;
REAL res, line_res;

REAL calc_sum(INT e1, INT e2, INT rel)
{
    REAL sum = 0;
    INT last1 = e1 * dimension;
    INT last2 = e2 * dimension;
    INT lastr = rel * dimension;
    for (INT ii = 0; ii < dimension; ii++)
	sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);
    return sum;
}

void gradient(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b)
{
    INT lasta1 = e1_a * dimension;
    INT lasta2 = e2_a * dimension;
    INT lastar = rel_a * dimension;
    INT lastb1 = e1_b * dimension;
    INT lastb2 = e2_b * dimension;
    INT lastbr = rel_b * dimension;
    for (INT ii = 0; ii < dimension; ii++) {
	REAL x;
	x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);
	if (x > 0)
	    x = -alpha;
	else
	    x = alpha;
	relationVecDao[lastar + ii] -= x;
	entityVecDao[lasta1 + ii] -= x;
	entityVecDao[lasta2 + ii] += x;
	x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);
	if (x > 0)
	    x = alpha;
	else
	    x = -alpha;
	relationVecDao[lastbr + ii] -= x;
	entityVecDao[lastb1 + ii] -= x;
	entityVecDao[lastb2 + ii] += x;
    }
}

void update(INT e1, INT e2, INT rel)
{
    REAL x1 = 0, x2 = 0, g1, g2;
    INT lasta1 = e1 * dimension;
    INT lasta2 = e2 * dimension;
    INT lastar = rel * dimension;
    for (INT ii = 0; ii < dimension; ii++) {
	x1 += entityVec[lasta1 + ii] * relationVec[lastar + ii];
	x2 += entityVec[lasta2 + ii] * relationVec[lastar + ii];
    }
    g1 = (1 - FastSigmoid(x1)) * alpha * con_lambda;
    g2 = -FastSigmoid(x2) * alpha * con_lambda;

    for (INT ii = 0; ii < dimension; ii++) {
	entityVecDao[lasta1 + ii] += g1 * relationVec[lastar + ii];
	relationVecDao[lastar + ii] += g1 * entityVec[lasta1 + ii];

	entityVecDao[lasta2 + ii] += g2 * relationVec[lastar + ii];
	relationVecDao[lastar + ii] += g2 * entityVec[lasta2 + ii];
    }
    line_res += log(FastSigmoid(x1)) * con_lambda;
    line_res += log(FastSigmoid(-x2)) * con_lambda;
}

void train_kb(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b, INT flag)
{
    REAL sum1 = calc_sum(e1_a, e2_a, rel_a);
    REAL sum2 = calc_sum(e1_b, e2_b, rel_b);
    if (sum1 + margin > sum2) {
	res += margin + sum1 - sum2;
	gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	if (flag == 1)
	    update(e1_a, e1_b, rel_a);
	else
	    update(e2_a, e2_b, rel_b);
    }
}

INT corrupt_head(INT id, INT h, INT r)
{
    INT lef, rig, mid, ll, rr;
    lef = lefHead[h] - 1;
    rig = rigHead[h];
    while (lef + 1 < rig) {
	mid = (lef + rig) >> 1;
	if (trainHead[mid].r >= r)
	    rig = mid;
	else
	    lef = mid;
    }
    ll = rig;
    lef = lefHead[h];
    rig = rigHead[h] + 1;
    while (lef + 1 < rig) {
	mid = (lef + rig) >> 1;
	if (trainHead[mid].r <= r)
	    lef = mid;
	else
	    rig = mid;
    }
    rr = lef;
    INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
    if (tmp < trainHead[ll].t)
	return tmp;
    if (tmp > trainHead[rr].t - rr + ll - 1)
	return tmp + rr - ll + 1;
    lef = ll, rig = rr + 1;
    while (lef + 1 < rig) {
	mid = (lef + rig) >> 1;
	if (trainHead[mid].t - mid + ll - 1 < tmp)
	    lef = mid;
	else
	    rig = mid;
    }
    return tmp + lef - ll + 1;
}

INT corrupt_tail(INT id, INT t, INT r)
{
    INT lef, rig, mid, ll, rr;
    lef = lefTail[t] - 1;
    rig = rigTail[t];
    while (lef + 1 < rig) {
	mid = (lef + rig) >> 1;
	if (trainTail[mid].r >= r)
	    rig = mid;
	else
	    lef = mid;
    }
    ll = rig;
    lef = lefTail[t];
    rig = rigTail[t] + 1;
    while (lef + 1 < rig) {
	mid = (lef + rig) >> 1;
	if (trainTail[mid].r <= r)
	    lef = mid;
	else
	    rig = mid;
    }
    rr = lef;
    INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
    if (tmp < trainTail[ll].h)
	return tmp;
    if (tmp > trainTail[rr].h - rr + ll - 1)
	return tmp + rr - ll + 1;
    lef = ll, rig = rr + 1;
    while (lef + 1 < rig) {
	mid = (lef + rig) >> 1;
	if (trainTail[mid].h - mid + ll - 1 < tmp)
	    lef = mid;
	else
	    rig = mid;
    }
    return tmp + lef - ll + 1;
}

void* trainMode(void* con)
{
    INT id, pr, i, j;
    id = (unsigned long long)(con);
    next_random[id] = rand();
    for (INT k = Batch; k >= 0; k--) {
	i = rand_max(id, Len);
	//		REAL ph = 1000*pow(1.0*freqEnt[trainList[i].h]/tripleTotal, 3.0/4);
	//		REAL pt = 1000*pow(1.0*freqEnt[trainList[i].t]/tripleTotal, 3.0/4);
	//		REAL p = 1000*pow(1.0*freqRel[trainList[i].r]/tripleTotal, 3.0/4);
	//		REAL tmp =rand() % 1000;
	//
	//		if(tmp < ph || tmp < pt || tmp < p)
	//			continue;
	if (bernFlag)
	    pr = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
	else
	    pr = 500;
	if (randd(id) % 1000 < pr) {
	    j = corrupt_head(id, trainList[i].h, trainList[i].r);
	    train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r, -1);
	} else {
	    j = corrupt_tail(id, trainList[i].t, trainList[i].r);
	    train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r, 1);
	}
	norm(relationVecDao + dimension * trainList[i].r);
	norm(entityVecDao + dimension * trainList[i].h);
	norm(entityVecDao + dimension * trainList[i].t);
	norm(entityVecDao + dimension * j);
    }
    pthread_exit(NULL);
}

void* train(void* con)
{
    Len = tripleTotal;
    Batch = 300;
    nbatches = Len / Batch / threads;
    next_random = (unsigned long long*)calloc(threads, sizeof(unsigned long long));
    REAL tmp = alpha;
    memcpy(relationVecDao, relationVec, dimension * relationTotal * sizeof(REAL));
    memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(REAL));
    for (INT epoch = 0; epoch < trainTimes; epoch++) {
	res = 0;
	line_res = 0;
	for (INT batch = 0; batch < nbatches; batch++) {
	    pthread_t* pt = (pthread_t*)malloc(threads * sizeof(pthread_t));
	    for (long a = 0; a < threads; a++)
		pthread_create(&pt[a], NULL, trainMode, (void*)a);
	    for (long a = 0; a < threads; a++)
		pthread_join(pt[a], NULL);
	    free(pt);
	    memcpy(relationVec, relationVecDao, dimension * relationTotal * sizeof(REAL));
	    memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(REAL));
	}
	printf("epoch %d lr %lf res %f line_res %f\n", epoch, alpha, res, line_res);
	alpha = tmp * (1 - 1.0 * epoch / trainTimes);
    }
}

/*
	Get the results of transE.
*/

void out_binary()
{
    INT len, tot;
    REAL* head;
    FILE* f2 = fopen((outPath + "relation2vec" + note + ".bin").c_str(), "wb");
    FILE* f3 = fopen((outPath + "entity2vec" + note + ".bin").c_str(), "wb");
    len = relationTotal * dimension;
    tot = 0;
    head = relationVec;
    while (tot < len) {
	INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f2);
	tot = tot + sum;
    }
    len = entityTotal * dimension;
    tot = 0;
    head = entityVec;
    while (tot < len) {
	INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f3);
	tot = tot + sum;
    }
    fclose(f2);
    fclose(f3);
}

void out()
{
    if (outBinaryFlag) {
	out_binary();
	return;
    }
    FILE* f2 = fopen((outPath + "relation2vec" + note + ".vec").c_str(), "w");
    FILE* f3 = fopen((outPath + "entity2vec" + note + ".vec").c_str(), "w");
    for (INT i = 0; i < relationTotal; i++) {
	INT last = dimension * i;
	for (INT ii = 0; ii < dimension; ii++)
	    fprintf(f2, "%.6f\t", relationVec[last + ii]);
	fprintf(f2, "\n");
    }
    for (INT i = 0; i < entityTotal; i++) {
	INT last = i * dimension;
	for (INT ii = 0; ii < dimension; ii++)
	    fprintf(f3, "%.6f\t", entityVec[last + ii]);
	fprintf(f3, "\n");
    }
    fclose(f2);
    fclose(f3);
}

/*
	Main function
*/

int ArgPos(char* str, int argc, char** argv)
{
    int a;
    for (a = 1; a < argc; a++)
	if (!strcmp(str, argv[a])) {
	    if (a == argc - 1) {
		printf("Argument missing for %s\n", str);
		exit(1);
	    }
	    return a;
	}
    return -1;
}

void setparameters(int argc, char** argv)
{
    int i;
    if ((i = ArgPos((char*)"-size", argc, argv)) > 0)
	dimension = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-input", argc, argv)) > 0)
	inPath = argv[i + 1];
    if ((i = ArgPos((char*)"-output", argc, argv)) > 0)
	outPath = argv[i + 1];
    if ((i = ArgPos((char*)"-load", argc, argv)) > 0)
	loadPath = argv[i + 1];
    if ((i = ArgPos((char*)"-thread", argc, argv)) > 0)
	threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-epochs", argc, argv)) > 0)
	trainTimes = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-nbatches", argc, argv)) > 0)
	nbatches = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-alpha", argc, argv)) > 0)
	alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char*)"-margin", argc, argv)) > 0)
	margin = atof(argv[i + 1]);
    if ((i = ArgPos((char*)"-load-binary", argc, argv)) > 0)
	loadBinaryFlag = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-out-binary", argc, argv)) > 0)
	outBinaryFlag = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-note", argc, argv)) > 0)
	note = argv[i + 1];
}

int main(int argc, char** argv)
{
    setparameters(argc, argv);
    init();
    if (loadPath != "")
	load();
    train(NULL);
    if (outPath != "")
	out();
    return 0;
}
