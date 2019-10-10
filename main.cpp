#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>

#include "md5.c";

char alphabet[94] =		  { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' , // 25
							'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', // 51
							'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', // 61
							'!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
							'{', '|', '}', '~' }; 

char * lowercase = &alphabet[0];
char * uppercase = &alphabet[25];
char * nums = &alphabet[51];
char * specials = &alphabet[61];
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
		uint8_t * hashedString  = md5(text, stringLength, hashPlaceholder);

		// check hash
		if (isHashEqual(hashedString, hash))
		{
			printf("%s\n", text);
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
// initialPermutation nastavi pocatecni kombinaci pismen, count urcuje pocet iteraci
char * bruteForceStep(int stringLength, char * alphabet, int alphabetSize, char * text, uint8_t hash[16], int * initialPermutation, double count)
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
		uint8_t * hashedString = md5(text, stringLength, hashPlaceholder);
		if (isHashEqual(hashedString, hash))
		{
			printf("%s\n", text);
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



int main()
{
	char str[40];
	scanf("%s", &str);
	char alpha[2] = { '0', '1' };
	uint8_t * hash = stringToHash(str);
	bruteForce(5, 5, alphabet, 62, hash);
	//bruteForce(1, 5, alpha, 2, hash);
	return 0; 
}