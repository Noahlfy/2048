#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "2048.h"


// Générer un nombre aléatoire : 2 ou 4

void initializeRandom() {
    srand(time(NULL)); 
}


int generate_random_number() {
    int random = rand() % 10; // Generate a random number between 0 and 9
    if (random == 0) {
        return 4;
    }
    return 2;
}

int getRandomNumber(int min, int max) {
    return rand() % (max - min + 1) + min;
}

// initialiser la mattrice

matrice *initialize_grid () {
    matrice *m = (matrice *) malloc(sizeof(matrice)); 
    if (m == NULL) {
        fprintf(stderr, "Erreur d'allocation de mémoire. \n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0 ; i < SIZE; i++) {
        for (int j = 0 ; j < SIZE; j++) {
            m->grid[i][j] = 0;
        }
    }
    m->score = 0;
    return m; 
}

void printMatrix(const matrice *mat) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%d ", mat->grid[i][j]);
        }
        printf("\n");
    }
}

void add_number (matrice *myGrid ) {
    int x, y;
    do {
        x = getRandomNumber(0, SIZE-1);
        y = getRandomNumber(0, SIZE-1);
    } while (myGrid->grid[x][y] != 0); 
    myGrid->grid[x][y] = generate_random_number();
}

void rotateRight(matrice* myGrid) {
    matrice* rotated = (matrice*)malloc(sizeof(matrice));
     if (rotated == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        exit(1);
    }
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            rotated->grid[j][SIZE-1-i] = myGrid->grid[i][j];
        }
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            myGrid->grid[i][j] = rotated->grid[i][j];
        }
    }
    free(rotated);
}

void transpose (matrice *myGrid, int direction) {
    for (int i=0; i<direction; i++) {
        rotateRight(myGrid);
    }
}


void play(matrice *myGrid, int direction) {

    // Liste pour ne pas additionner deux fois sur une même case
    matrice *check = initialize_grid();

    transpose(myGrid, direction);
    for (int i = 1; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (myGrid->grid[i][j] != 0) {
                int k = i - 1;

                while (k >= 0 && (myGrid->grid[k][j] == 0 || myGrid->grid[k][j] == myGrid->grid[k+1][j]) && check->grid[k][j] == 0 ) {                    

                    if (myGrid->grid[k][j] == myGrid->grid[k+1][j]) {
                        for (int l = k; l>=0; l--) {
                            check->grid[l][j] = 1;
                        }
                        myGrid->score += 2*myGrid->grid[k][j];
                    }
                    myGrid->grid[k][j] += myGrid->grid[k+1][j];
                    myGrid->grid[k+1][j] = 0;

                    k--;
                }
            }
        }
    }

    if (direction == RIGHT) {
        transpose(myGrid, 1);
    } else if (direction == LEFT) {
        transpose(myGrid, 3);
    } else {
        transpose(myGrid, direction);
    }
    add_number (myGrid);
}

int max (matrice *myGrid) {
    int maximum = 0;
    for (int i=0; i<SIZE; i++) {
        for (int j=0; j<SIZE; j++) {
            if (myGrid->grid[i][j] > maximum) {
                maximum = myGrid->grid[i][j];
            }
        }
    }
    return maximum;
}

int directionFromString(char *input) {
    int direction = -1;
    if (strcmp(input, "LEFT") == 0) {
        direction = LEFT;
    } else if (strcmp(input, "RIGHT") == 0) {
        direction = RIGHT;
    } else if (strcmp(input, "UP") == 0) {
        direction = UP;
    } else if (strcmp(input, "DOWN") == 0) {
        direction = DOWN;
    } else {
        printf("Direction non reconnue. Utilisez LEFT, RIGHT, UP, DOWN.\n");
    }
    return direction;
}


int gameFinished (matrice *myGrid) {
    matrice myGridCopy = *myGrid; 
    
    for (int i=0; i<SIZE; i++) {
        play(&myGridCopy, i);
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (myGridCopy.grid[i][j] != myGrid->grid[i][j]) {
                return 0;
            }
        }
    }
    printf("La partie est terminée; \n");
    return 1;
}
