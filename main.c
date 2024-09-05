// Créer un 2048 en C : 

// La structure : 4x4 

// Développer une IA : Commencer par un réseau de neurones puis voir des techniques d'algo

// Ecrire une page en JS pour pouvoir y jouer

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "AI.h"
#include "2048.h"

int main() {

    initializeRandom();

    // Initialisation de la grille
    
    matrice *myGrid = initialize_grid();
    add_number(myGrid);

    for (int i = 0; i<SIZE; i++) {
        for (int j = 0; j<SIZE; j++) {
            printf(" %d ", myGrid->grid[i][j]);
        }
        printf("\n");
    }

    // Jouer tant que la partie n'est pas finie
    do
    {   
        int direction = -1;
        while (direction == -1) {
            char input[10]; // Buffer pour la direction
            printf("Entrez la direction (LEFT, RIGHT, UP, DOWN): \n");
            scanf("%9s", input);
            direction = directionFromString(input);
        }

        play(myGrid, direction);
        printMatrix(myGrid);

    } while (gameFinished(myGrid) == 0);
    
    free(myGrid);
}