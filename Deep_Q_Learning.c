#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <math.h>

#include "AI.h"
#include "2048.h"
#include "Deep_Q_Learning.h"


float epsilon = 1.0f; 

Tensor *Copy_Tensor(Tensor *t) {
    Tensor *copy = (Tensor*)malloc(sizeof(Tensor));
    if (copy == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour la copie du Tensor.\n");
        exit(EXIT_FAILURE);
    }

    copy->ndim = t->ndim;
    copy->size = t->size;

    copy->shape = (size_t*)malloc(copy->ndim * sizeof(size_t));
    if (copy->shape == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les dimensions du Tensor copié.\n");
        free(copy);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < copy->ndim; i++) {
        copy->shape[i] = t->shape[i];
    }

    copy->data = (float*)malloc(copy->size * sizeof(float));
    if (copy->data == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les données du Tensor copié.\n");
        free(copy->shape);
        free(copy);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < copy->size; i++) {
        copy->data[i] = t->data[i];
    }

    return copy;
}

Filter *copy_filter(Filter *f) {
    Filter *copy = (Filter*)malloc(sizeof(Filter));
    if (copy == NULL) {
        fprintf(stderr, "Erreur d'allocation de mémoire pour la copie des filtres. \n");
        exit(EXIT_FAILURE);
    }
    copy->filter_size = f->filter_size;
    copy->input_channels = f->input_channels;
    copy->output_channels = f->output_channels;

    for (size_t i=0; i < f->filter_size; i++) {
        for (size_t j=0; j < f->filter_size; j++) {
            for( size_t k=0; k < f->input_channels; k++) {
                copy->data[i][j][k] = f->data[i][j][k];
            }
        }
    }
    
}

Layer *Copy_Layer(Layer *l) {
    Layer *copy = (Layer*)malloc(sizeof(Layer));
    if (copy == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour la copie du Layer.\n");
        exit(EXIT_FAILURE);
    }
    copy->type = l->type;
    
    switch (l->type) {
        case DENSE : 
            copy->dense.input_dim = l->dense.input_dim;
            copy->dense.output_dim = l->dense.output_dim;
            copy->dense.activation = l->dense.activation;

            copy->dense.weights = *Copy_Tensor(&l->dense.weights);
            copy->dense.biases = *Copy_Tensor(&l->dense.biases);
            copy->dense.output_data = *Copy_Tensor(&l->dense.output_data);
            break ; 

        case CONV : 
            copy->conv.activation = l->conv.activation ;
            copy->conv.input_channels = l->conv.input_channels ;
            copy->conv.output_channels = l->conv.output_channels ;
            copy->conv.input_height = l->conv.input_height ;
            copy->conv.input_width = l->conv.input_width ;
            copy->conv.padding = l->conv.padding ;
            copy->conv.stride = l->conv.stride ;

            copy->conv.output_data = *Copy_tensor(&l->conv.output_data);
            copy->conv.biases = *Copy_Tensor(&l->conv.biases);

            for (size_t i=0; i < l->conv.output_channels; i++) {
                copy->conv.filters[i] = *copy_filter(&l->conv.filters[i]);
            }
                        
            break;
        
        default : 
            fprintf(stderr, 'La couche est ma définie.');
            break;
    }
 

    return copy;
}



NeuralNetwork *copy_network(NeuralNetwork *nn) {
    NeuralNetwork *copy = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (copy == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour la copie du NeuralNetwork.\n");
        exit(EXIT_FAILURE);
    }

    copy->num_layers = nn->num_layers;
    copy->layers = (Layer*)malloc(copy->num_layers * sizeof(Layer));
    if (copy->layers == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les couches du NeuralNetwork copié.\n");
        free(copy);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < copy->num_layers; i++) {
        copy->layers[i] = *Copy_Layer(&nn->layers[i]);
    }
    return copy;
}


NeuralNetwork *Q_Network () {
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (nn == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour le Q-Network.\n");
        exit(EXIT_FAILURE);
    }

    Layer layer1 = dense_layer(INPUT_DIM, 150, relu);
    Layer layer2 = dense_layer(150, 200, relu);
    Layer layer3 = dense_layer(200, OUPUT_DIM, relu);

    Layer layers[] = {layer1, layer2, layer3};

    *nn = initialize_NeuralNetwork(layers, 3);

    return nn;
}

// Algorithme Epsilon greedy

// Fonction pour trouver l'indice de la valeur maximale dans un tableau
int argmax(float *array, size_t size) {
    int max_idx = 0;
    float max_val = -INFINITY;
    for (size_t i = 0; i < size; ++i) {
        if (array[i] > max_val) {
            max_val = array[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int chose_action(matrice *state, NeuralNetwork *Q_Network) {
    srand((unsigned int)time(NULL)); 

    size_t input_shape[] = {SIZE * SIZE}; 
    Tensor input = initialize_tensor(input_shape, 1);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            input.data[i * SIZE + j] = (float)state->grid[i][j];
        }
    }

    int action;
    if ((float)rand() / RAND_MAX < epsilon) {
        // Exploration : choisir une action aléatoire
        action = rand() % OUPUT_DIM; 
    } else {
        // Exploitation : choisir l'action avec la valeur Q maximale
        Tensor Q_values = initialize_tensor((size_t[]){OUPUT_DIM}, 1);
        forward_pass_network(Q_Network, &input, &Q_values);
        action = argmax(Q_values.data, Q_values.size);
        free_tensor(&Q_values);
    }
    free_tensor(&input);
    return action;
}

float max (float i, float j) {
    if (i > j) {
        return i;
    }
    return j;
}

void update_epsilon() {
    if (epsilon > EPSILON_MIN) {
        epsilon *= EPSILON_DECAY;
        epsilon = max(epsilon, EPSILON_MIN);
    }
}

// Calculer les valeurs de Q à l'aide de Q_Network
void predict_Q (matrice *myGrid, NeuralNetwork *Q_Network, Tensor *Q_values) {

    size_t input_shape[] = {SIZE * SIZE};
    Tensor input = initialize_tensor(input_shape, 1);
    
    for (int i=0; i < SIZE; i++) {
        for (int j=0; j < SIZE; j++) {
            input.data[i * SIZE +j] = (float)myGrid->grid[i][j];
        }
    }
    forward_pass_network(Q_Network, &input, Q_values);
    // Les valeur sont stockées dans Q_values->data

    free_tensor(&input);
}

void target_Q (matrice *state, NeuralNetwork *Q_Network, NeuralNetwork *Target_Network, float gamma, int direction) {
    
    matrice current_state = *state;
    play(state, direction);
    int reward = state->score - current_state.score;

    size_t input_shape[] = {SIZE * SIZE};
    Tensor input = initialize_tensor(input_shape, 1);
    for (int i=0; i < SIZE; i++) {
        for (int j=0; j < SIZE; j++) {
            input.data[i * SIZE +j] = (float)state->grid[i][j];
        }
    }
    size_t Q_target_shape[] = {SIZE};
    Tensor Q_target = initialize_tensor(Q_target_shape, 1);
    forward_pass_network(Target_Network, &input, &Q_target);

    float Max_Q_target = -INFINITY;
    for (int i = 0; i < OUPUT_DIM; i++) {
        if (Q_target.data[i] > Max_Q_target) {
            Max_Q_target = Q_target.data[i];
        }
    }

    free_tensor(&input);
    free_tensor(&Q_target);

    return reward + gamma * Max_Q_target;

}

float loss(float Q_predict, float Q_target) {
    return 0.5 * (Q_predict - Q_target) * (Q_predict - Q_target);
}

void update_weights (NeuralNetwork *nn, float Q_target, float Q_predict, float alpha) {

    // We use the backpropagation for the Q-learning
    
    float initial_gradient_loss = Q_target - Q_predict;
    Tensor activation_gradient;
    Tensor gradient;

    // For the output layer
    switch (nn->layers[nn->num_layers - 1].type)
    {
    case DENSE:
        DenseLayer *output_layer = &nn->layers[nn->num_layers - 1].dense;

        activation_gradient = gradient_activation(output_layer->activation, &output_layer->output_data);

        // gradient = activation_gradient * initial_gradient_loss
        size_t gradient_shape[] = activation_gradient.shape;
        gradient = initialize_tensor(gradient_shape, activation_gradient.ndim);
        
        for (size_t i=0; i < activation_gradient.size; i++) {
            gradient.data[i] = activation_gradient.data[i] * initial_gradient_loss;
        }

        // Tensor weight_grad = gradient * nn->layers[nn->num_layers].dense.output_data ;
        Tensor weight_grad = initialize_tensor((size_t[]){output_layer->input_dim, output_layer->output_dim}, 2);
        for (size_t i=0; i < output_layer->input_dim; i++ ) {
            for (size_t j=0; j < output_layer->output_dim; j++) {
                weight_grad.data[i *output_layer->input_dim + j] = gradient.data[j] * nn->layers[nn->num_layers - 2].dense.output_data.data[i];
            }
        }

        Tensor biases_grad = initialize_tensor((size_t[]){output_layer->output_dim}, 1);
        for (size_t i=0; i < output_layer->output_dim; i++) {
            biases_grad.data[i] = gradient.data[i];
        }

        // Update weights and biases
        for (size_t i = 0; i < output_layer->weights.size; i++) {
            output_layer->weights.data[i] -= alpha * weight_grad.data[i];
        }
        for (size_t j = 0; j < output_layer->biases.size; j++) {
            output_layer->biases.data[j] -= alpha * biases_grad.data[j];
        }

        // Free temporary tensors
        free_tensor(&weight_grad);
        free_tensor(&biases_grad);
        free_tensor(&gradient);
        free_tensor(&activation_gradient);
        break;
        break;
    
    default :
        break;
    }

    // General case
    for (int i = nn->num_layers - 2; i >= 0 ; i--) {

        switch (nn->layers[i].type)
        {
        case DENSE :
            DenseLayer *layer = &nn->layers[i].dense;
            DenseLayer *next_layer = &nn->layers[i-1].dense;

            activation_gradient = gradient_activation(layer->activation, &layer->output_data);

            // gradient = activation_gradient * gradient

            Tensor new_gradient = initialize_tensor((size_t[]){layer->output_dim}, 1);
            for (size_t j=0; j < layer->output_dim; j++) {

                // new_gradient.data[j] *= activation_gradient 
                for (size_t l=0; l < layer->output)
            }
            break;
        
        default:
            break;
        }
        // Tensor Gradient_loss = gradient_activation(nn->layers[i].activation, ); 

        // for (int j=0; j < nn->layers[i].weights.size; j++) {
        //     nn->layers[i].weights.data[j] -= alpha * Gradient_loss;
        // }    

        // for (int k=0; k < nn->layers[i].biases.size; k++) {

        // } 
    }
}


void train_model (matrice *state, int epochs) {
    int direction = chose_action(state, Q_Network); 
    update_epsilon();

    Tensor *Q_value;
    float Q_predict = Q_values->data[direction];
}

// Calculer les valeurs cibles en utilisant le target network.
// Calculer la perte et ajuster les poids du Q-network.
