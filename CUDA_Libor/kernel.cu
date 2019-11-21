#define _CRT_SECURE_NO_WARNINGS
#define THREADS 128 /// 128
#define BLOCKS 32 /// 32

#define THRESHOLD 1000

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "md5.c"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>


char tnumbers[11] = "0123456789";
char tlowercase[27] = "abcdefghijklmnopqrstuvwxyz";
char tuppercase[27] = "ABCDEFGHIJKLMNOPQRTSUVWXYZ";
char tlowerupper[53] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRTSUVWXYZ";
char tloweruppernums[63] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRTSUVWXYZ0123456789";
char allchars[95] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRTSUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
char * alph[6] = { &tnumbers[0], &tlowercase[0], &tuppercase[0], &tlowerupper[0], &tloweruppernums[0], allchars };
char sizes[6] = { 10, 26, 26, 52, 62, 94 };
int limits[6] = { 19, 13, 13, 11, 10, 9 }; // max word length

char * valuePlaceholder;

__constant__ char alphabetGPU[95]; 
__constant__ uint32_t hashGPU[4]; 


#pragma region GPU
__device__ void isHashEqualNewDevice(uint32_t * hash1, uint32_t * hash2, bool * ret)
{
	*ret = true;
	for (int i = 0; i < 4; i++)
	{
		if (hash1[i] != hash2[i])
		{
			*ret = false;
			return;
		}
	}
}

__device__ void * bruteForceStepDevice(int stringLength, int alphabetSize, char * text, int * initialPermutation, uint64_t count, char * valuePlaceholder)
{
	// pouzit uint32_t

	uint32_t hashPlaceHolderNew[4];

	uint32_t hashLocal[4]; 
	memcpy(hashLocal, hashGPU, 4 * sizeof(uint32_t));
	for (int i = 0; i < stringLength; i++)
	{
		// nastaveni stringu do pocatecniho stavu
		text[i] = alphabetGPU[initialPermutation[i]];
	}


	text[stringLength] = 0;
	bool overflow = false;
	uint64_t localCount = 0;
	bool retTmp = false;
	uint8_t msg[200]; 
	while (!overflow)
	{
		
		md5Device(text, stringLength, hashPlaceHolderNew, msg);
		isHashEqualNewDevice(hashPlaceHolderNew, hashLocal, &retTmp);
		if (retTmp)
		{
			memcpy(valuePlaceholder, text, stringLength * sizeof(char)); 
			//free(hashedString);
			return; 
		}
		initialPermutation[0]++;
		initialPermutation[0] %= alphabetSize;
		text[0] = alphabetGPU[initialPermutation[0]];
		if (initialPermutation[0] == 0)
		{
			for (int i = 1; i < stringLength; i++)
			{
				// carry chain
				initialPermutation[i]++;
				initialPermutation[i] %= alphabetSize;
				text[i] = alphabetGPU[initialPermutation[i]];
				if (initialPermutation[i] != 0)
					break;
				else
					if (i == stringLength)
						overflow = true;
			}
		}

		localCount++;
		if (localCount >= count)
			break;
		if (localCount % THRESHOLD == 0)
		{
			if (valuePlaceholder[0] != 0)
				return; 
		}
	}
	// hash nenalezen
	return NULL;
}

__global__ void bruteForceDevice(int len, uint64_t total, int alphabetSize, char * valuePlaceholder)
{
	cudaError_t cudaStatus;
	char * text;
	int * initialPermutation; 
	int threadCount = blockDim.x;
	int blockCount = gridDim.x;
	double tmp = ((double)total / blockCount);
	uint64_t blockWork = (int)ceil(tmp);
	tmp = (double)total / (blockCount*threadCount);
	uint64_t threadWork = (int)ceil(tmp);
	uint64_t start = blockIdx.x * blockWork + threadIdx.x * threadWork;
	

	text = (char*)malloc(len + 1); 

	if (text == 0)
	{
		printf("bruteForceDevice, nelze alokovat pamet pro text\n");
		return;
	}

	//char * value = bruteForceStepRec(0, i, alphabet, alphabetSize, text, hash);


	initialPermutation = (int*)malloc(len * sizeof(int));
	if (initialPermutation == NULL) {
		printf("bruteForceDevice, nelze alokovat pamet\n");
		return;
	}
	

	for (int i = 0; i < len; i++)
	{

		initialPermutation[i] = start % alphabetSize;
		start /= alphabetSize; 
	}

	bruteForceStepDevice(len, alphabetSize, text, initialPermutation, threadWork, valuePlaceholder);

	free(text);
	free(initialPermutation);
}

__host__ char * cudaBruteForceStart(int minLength, int maxLength, char * alphabet, int alphabetSize, uint32_t hash[4])
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpyToSymbol((const void *)alphabetGPU, alphabet, alphabetSize, 0, cudaMemcpyHostToDevice);

	char * originalValue = NULL;

	if (cudaStatus != cudaSuccess) {
		printf("cudaBruteForceStart, nelze alokovat symbol alphabetGPU\n");
		return originalValue;
	}

	cudaStatus = cudaMemcpyToSymbol((const void *)hashGPU, hash, sizeof(uint32_t) * 4, 0, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaBruteForceStart, nelze alokovat symbol hash\n");
		return originalValue;
	}

	for (int i = minLength; i <= maxLength; i++)
	{
		originalValue = NULL; 
		cudaStatus = cudaMalloc((void**)&valuePlaceholder, (size_t)(i * sizeof(char) + 1));
		if (cudaStatus != cudaSuccess) {
			printf("cudaBruteForceStart, nelze alokovat pamet\n");
			return originalValue;
		}

		cudaStatus = cudaMemset(valuePlaceholder, 0, (size_t)i + 1); 
		if (cudaStatus != cudaSuccess) {
			printf("cudaBruteForceStart, nelze nastavit pamet\n");
			return originalValue;
		}
		uint64_t total = pow(alphabetSize, i);
		bruteForceDevice << <BLOCKS, THREADS >> >(i, total, alphabetSize, valuePlaceholder);
		originalValue = (char *)malloc(i * sizeof(char) + 1);
		cudaDeviceSynchronize();

		cudaStatus = cudaMemcpy(originalValue, valuePlaceholder, i, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("cudaBruteForceStart, nelze zkopirovat pamet\n");
			free(originalValue);
			originalValue = NULL; 
			return originalValue;
		}
		cudaFree(valuePlaceholder);

		if (originalValue[0] != 0)
			break;

		free(originalValue);
	}
	return originalValue; 
}
#pragma endregion GPU

#pragma region CPU
void isHashEqualNew(uint32_t * hash1, uint32_t * hash2, bool * ret)
{
	*ret = true; 
	for (int i = 0; i < 4; i++)
	{
		if (hash1[i] != hash2[i])
		{
			*ret = false;
			return; 
		}
	}
}

// nerekurzivni volani - idealni pro GPU?
// initialPermutation nastavi pocatecni kombinaci pismen, toto se zjisti tak, ze se vezme pocatecni hodnota pro dane vlakno a postupne se vymoduli/vydeli toto cislo a ziska se tim permutace. 
//Bude to fungovat podobne jako kdyz se napr. desitkove cislo prevadi na sestnactkove, count urcuje pocet iteraci
char * bruteForceStepNew(int stringLength, char * alphabet, int alphabetSize, char * text, uint32_t hash[4], int * initialPermutation, uint64_t count)
{
	// pouzit uint32_t
	uint32_t hashPlaceHolderNew[4]; 
	for (int i = 0; i < stringLength; i++)
	{
		// nastaveni stringu do pocatecniho stavu
		text[i] = alphabet[initialPermutation[i]];
	}
	text[stringLength] = 0;
	bool overflow = false;
	double localCount = 0;
	bool retTmp = false; 

	while (!overflow)
	{
		md5(text, stringLength, hashPlaceHolderNew);
		isHashEqualNew(hashPlaceHolderNew, hash, &retTmp);
		if (retTmp)
		{
			//free(hashedString);
			return text;
		}
		initialPermutation[0]++;
		initialPermutation[0] %= alphabetSize;
		text[0] = alphabet[initialPermutation[0]];
		if (initialPermutation[0] == 0)
		{
			for (int i = 1; i <= stringLength; i++)
			{
				// carry chain
				initialPermutation[i]++;
				initialPermutation[i] %= alphabetSize;
				text[i] = alphabet[initialPermutation[i]];
				if (initialPermutation[i] != 0)
					break;
				else
					if (i == stringLength)
						overflow = true;
			}
		}

		localCount++;
		if (localCount >= count)
			break;
	}
	// hash nenalezen
	return NULL;
}

char * bruteForceNew(int minLength, int maxLength, char * alphabet, int alphabetSize, uint32_t hash[4])
{
	for (int i = minLength; i <= maxLength; i++)
	{
		char * text = (char *)malloc((i + 1) * sizeof(char));
		//char * value = bruteForceStepRec(0, i, alphabet, alphabetSize, text, hash);
		int * initialPermutation = (int*)calloc(sizeof(int), i);
		uint64_t count = pow(alphabetSize, i);
		char * value = bruteForceStepNew(i, alphabet, alphabetSize, text, hash, initialPermutation, count);
		if (value != NULL)
			return value;
		free(text);
	}

	return NULL;
}


void badUsage()
{
	printf("Usage: PATH_TO_PROGRAM mode [path_to_dictionary | alphabet] hash [min_length] [max_length]\n\n");
	printf("MODE:\n");
	printf("0 - Dictionary attack\n");
	printf("    Usage: PATH_TO_PROGRAM 0 path_to_dictionary hash\n");
	printf("1 - Brute-force attack\n");
	printf("    Usage: PATH_TO_PROGRAM 1 alphabet hash min_length max_length\n\n");
	printf("2 - Brute-force attack GPU\n");
	printf("    Usage: PATH_TO_PROGRAM 1 alphabet hash min_length max_length\n\n");
	printf("ALPHABET:\n");
	printf("0 - numbers only\n");
	printf("1 - lower case\n");
	printf("2 - upper case\n");
	printf("3 - lower+upper case\n");
	printf("4 - lower+upper case + numbers\n");
	printf("5 - all characters\n");
}

uint32_t * stringToHashNew(char * hashString)
{
	uint8_t * hashArray = (uint8_t *)malloc(16 * sizeof(uint8_t));
	uint32_t * hashNew = (uint32_t *)calloc(4, sizeof(uint32_t));
	for (int i = 0; i < 16; i++)
	{
		uint8_t hiNibble = (uint8_t)tolower(hashString[2 * i]);
		hiNibble > 57 ? hiNibble -= 87 : hiNibble -= 48;
		uint8_t loNibble = (uint8_t)tolower(hashString[2 * i + 1]);
		loNibble > 57 ? loNibble -= 87 : loNibble -= 48;
		hashArray[i] = hiNibble << 4 | loNibble;
	}
	for (int i = 0; i < 16; i++)
	{
		uint32_t tmp = hashArray[i] << ((i % 4) * 8); 
		hashNew[i / 4] |= tmp; 
	}
	free(hashArray);
	return hashNew;
}

#pragma endregion CPU

int main(int argc, char *argv[])
{
 
	uint32_t *hash;
	char *originalString;
	int mode = -1;
	int alphabetMode = -1;
	char *_alphabet;
	int alphabetLen;

	if (argc < 4) {
		badUsage();
		return -1;
	}

	mode = atoi(argv[1]);
	hash = stringToHashNew(argv[3]);

	if (mode == 0) {
		//originalString = dictionaryAttack(argv[2], hash);
		if (originalString == NULL) {
			printf("No matches!\n");
			return 0;
		}
		else {
			printf("%s\n", originalString);
			return 0;
		}
	}
	if (mode == 1) {
		if (argc < 6) {
			badUsage();
			return -1;
		}

		int minLenght = -1, maxLength = -1;
		alphabetMode = atoi(argv[2]);
		minLenght = atoi(argv[4]);
		maxLength = atoi(argv[5]);

		if (minLenght > maxLength) {
			printf("Minimum length must be less than or equal to the maximum length.\n");
			return -1;
		}

		if (alphabetMode >= 0 && alphabetMode <= 5) {
			_alphabet = alph[alphabetMode];
			alphabetLen = sizes[alphabetMode];
		}
		else {
			badUsage();
			return -1;
		}

		originalString = cudaBruteForceStart(minLenght, maxLength, _alphabet, alphabetLen, hash);
		//originalString = bruteForceNew(minLenght, maxLength, _alphabet, alphabetLen, hash);
		if (originalString == NULL) {
			printf("No matches!\n");
			return 0;
		}
		else {
			printf("%s\n", originalString);
			return 0;
		}
		
	}
	else if (mode == 2)
	{
		if (argc < 6) {
			badUsage();
			return -1;
		}

		int minLenght = -1, maxLength = -1;
		alphabetMode = atoi(argv[2]);
		minLenght = atoi(argv[4]);
		maxLength = atoi(argv[5]);

		if (minLenght > maxLength) {
			printf("Minimum length must be less than or equal to the maximum length.\n");
			return -1;
		}

		if (alphabetMode >= 0 && alphabetMode <= 5) {
			_alphabet = alph[alphabetMode];
			alphabetLen = sizes[alphabetMode];
		}
		else {
			badUsage();
			return -1;
		}

		originalString = cudaBruteForceStart(minLenght, maxLength, _alphabet, alphabetLen, hash);
		//originalString = bruteForceNew(minLenght, maxLength, _alphabet, alphabetLen, hash);
		if (originalString[0] == 0) {
			printf("No matches!\n");
			return 0;
		}
		else {
			printf("%s\n", originalString);
			return 0;
		}
	}

	return 0;
}


