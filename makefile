# Nom de l'exécutable
TARGET = 2048

# Compilateur et options de compilation
CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c99

# Liste des fichiers source
SRCS = main.c 2048.c

# Liste des fichiers objets (générés à partir des fichiers source)
OBJS = $(SRCS:.c=.o)

# Règle par défaut pour compiler l'exécutable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Règle pour compiler les fichiers objets
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Règle pour nettoyer les fichiers générés
.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
