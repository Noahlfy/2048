#ifndef MATRICE_H
#define MATRICE_H

#define SIZE 4
#define UP 0
#define LEFT 1
#define DOWN 2
#define RIGHT 3



typedef struct {
    int grid[SIZE][SIZE];
    int score;
} matrice;

// Déclarations des fonctions
void initializeRandom();
int generate_random_number();
int getRandomNumber(int min, int max);
matrice *initialize_grid();
void printMatrix(const matrice *mat);
void add_number(matrice *myGrid);
void rotateRight(matrice* myGrid);
void transpose (matrice *myGrid, int direction);
void play(matrice *myGrid, int direction);
int max (matrice *myGrid);
int directionFromString(char *input);
int gameFinished (matrice *myGrid);

#endif 
