#define _CRT_SECURE_NO_WARNINGS
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>

#include "md5.c"

char alphabet[94] =		  { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' , // 25
							'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', // 51
							'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', // 61
							'!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
							'{', '|', '}', '~' }; 

char * lowercase = &alphabet[0];
char * uppercase = &alphabet[26];
char * nums = &alphabet[52];
char * specials = &alphabet[62];
uint8_t hashPlaceholder[16]; 

bool isHashEqual(uint8_t * hash1, uint8_t * hash2)
{
	for (int i = 0; i < 16; i++)
	{
		if (hash1[i] != hash2[i])
			return false;
	}
	return true;
}

char * bruteForceStepRec(int step, int stringLength, char * alphabet, int alphabetSize, char * text, uint8_t hash[16])
{
	if (step == stringLength)
	{
		text[step] = 0;
		md5(text, stringLength, hashPlaceholder);

		// check hash
		if (isHashEqual(hashPlaceholder, hash))
		{
			//free(hashedString);
			return text; 
		}
		//free(hashedString);
		

		return NULL; 
	}
	for (int i = 0; i < alphabetSize; i++)
	{
		text[step] = alphabet[i];
		char * ret = bruteForceStepRec(step + 1, stringLength, alphabet, alphabetSize, text, hash);
		if (ret != NULL)
			return ret; 
	}
	return NULL; 
}


// nerekurzivni volani - idealni pro GPU?
// initialPermutation nastavi pocatecni kombinaci pismen, toto se zjisti tak, ze se vezme pocatecni hodnota pro dane vlakno a postupne se vymoduli/vydeli toto cislo a ziska se tim permutace. 
//Bude to fungovat podobne jako kdy� se napr. desitkove cislo prevadi na sestnactkove, count urcuje pocet iteraci
char * bruteForceStep(int stringLength, char * alphabet, int alphabetSize, char * text, uint8_t hash[16], int * initialPermutation, uint64_t count)
{
	for (int i = 0; i < stringLength; i++)
	{
		// nastaveni stringu do pocatecniho stavu
		text[i] = alphabet[initialPermutation[i]];
	}
	text[stringLength] = 0; 
	bool overflow = false;
	double localCount = 0;
	
	while (!overflow)
	{
		md5(text, stringLength, hashPlaceholder);
		if (isHashEqual(hashPlaceholder, hash))
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

char * bruteForce(int minLength, int maxLength, char * alphabet, int alphabetSize, uint8_t hash[16])
{
	for (int i = minLength; i <= maxLength; i++)
	{
		char * text = (char *)malloc((i + 1) * sizeof(char));
		//char * value = bruteForceStepRec(0, i, alphabet, alphabetSize, text, hash);
		int * initialPermutation = (int*)calloc(sizeof(int), i);
		uint64_t count = pow(alphabetSize, i); 
		char * value = bruteForceStep(i, alphabet, alphabetSize, text, hash,initialPermutation,count);
		if (value != NULL)
			return value;
		free(text);
	}

	return NULL; 
}

char * dictionaryAttack(char * dictionaryFile, uint8_t hash[16])
{
	FILE * fp = fopen(dictionaryFile, "r");

	if (fp == NULL)
	{
		printf("Cant open the dictionary");
		return NULL; 
	}
	char * word = (char*)malloc(255);
	while (true)
	{
		if (fscanf(fp, "%s", word) == EOF)
			break; 
		int len = strlen(word);
		md5(word, len, hashPlaceholder);
		if (isHashEqual(hashPlaceholder, hash))
		{
			//free(hashedString);
			fclose(fp);
			return word;
		}
	}

	free(word);
	fclose(fp);
	return NULL; 

}

uint8_t * stringToHash(char * hashString)
{
	uint8_t * hashArray = (uint8_t *)malloc(16 * sizeof(uint8_t));
	for (int i = 0; i < 16; i++)
	{
		uint8_t hiNibble = (uint8_t)tolower(hashString[2 * i]);
		hiNibble > 57 ? hiNibble -= 87 : hiNibble -= 48;
		uint8_t loNibble = (uint8_t)tolower(hashString[2 * i + 1]);
		loNibble > 57 ? loNibble -= 87 : loNibble -= 48;
		hashArray[i] = hiNibble << 4 | loNibble; 
	}
	return hashArray;
}

void badUsage()
{
	printf("Usage: PATH_TO_PROGRAM mode [path_to_dictionary | alphabet] hash [min_length] [max_length]\n\n");
	printf("MODE:\n");
	printf("0 - Dictionary attack\n");
	printf("    Usage: PATH_TO_PROGRAM 0 path_to_dictionary hash\n"); 
	printf("1 - Brute-force attack\n");
	printf("    Usage: PATH_TO_PROGRAM 1 alphabet hash min_length max_length\n\n"); 
	printf("ALPHABET:\n");
	printf("0 - numbers only\n");
	printf("1 - lower case\n");
	printf("2 - upper case\n");
	printf("3 - lower+upper case\n");
	printf("4 - lower+upper case + numbers\n");
	printf("5 - all characters\n");
}

int main(int argc, char *argv[])
{
	
	//char str[40];
	//scanf("%s", &str);
	//char alpha[2] = { '0', '1' };
	//uint8_t * hash = stringToHash(str);
	//dictionaryAttack("E:\\Documents\\Visual Studio 2017\\Projects\\HashSekv\\HashSekv\\realhuman_phill.txt", hash);
	//bruteForce(4, 4, alphabet, 62, hash);
	//bruteForce(1, 5, alpha, 2, hash);
	uint8_t * hash;
	char * originalString; 
	int mode = -1;
	int alphabetMode = -1; 
	char * _alphabet; 
	int alphabetLen; 
	if (argc < 4)
	{
		badUsage(); 
		return -1; 
	}
	mode = atoi(argv[1]);
	hash = stringToHash(argv[3]);
	if (mode == 0)
	{
		originalString = dictionaryAttack(argv[2], hash);
		if (originalString == NULL)
		{
			printf("No matches!\n"); 
			return 0;
		}
		else
		{
			printf("%s\n", originalString); 
			return 0; 
		}
	}
	else if (mode == 1)
	{
		if (argc < 6)
		{
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

		if (alphabetMode == 0)
		{
			_alphabet = nums;
			alphabetLen = 10; 
		}
		else if (alphabetMode == 1)
		{
			_alphabet = lowercase;
			alphabetLen = 26;
		}
		else if (alphabetMode == 2)
		{
			_alphabet = uppercase;
			alphabetLen = 26;
		}
		else if (alphabetMode == 3)
		{
			_alphabet = alphabet;
			alphabetLen = 52;
		}
		else if (alphabetMode == 4)
		{
			_alphabet = alphabet;
			alphabetLen = 62;
		}
		else if (alphabetMode == 5)
		{
			_alphabet = alphabet;
			alphabetLen = 94;
		}
		else
		{
			badUsage();
			return -1;
		}

		originalString = bruteForce(minLenght, maxLength, _alphabet, alphabetLen, hash);
		if (originalString == NULL)
		{
			printf("No matches!\n");
			return 0;
		}
		else
		{
			printf("%s\n", originalString);
			return 0;
		}

	}

	return 0; 
}