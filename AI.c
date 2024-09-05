#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <math.h>

#include <AI.h>

Tensor initialize_tensor(size_t *shape, size_t ndim)
{
    Tensor t;
    t.ndim = ndim;

    t.shape = (size_t *)malloc(ndim * (sizeof(size_t)));
    if (t.shape == NULL)
    {
        fprintf(stderr, "Erreur d'allocation de mémoire pour les dimensions du tenseur.\n");
        exit(EXIT_FAILURE);
    }

    t.size = 1;
    for (int i = 0; i < ndim; i++)
    {
        t.shape[i] = shape[i];
        t.size *= shape[i];
    }

    t.data = (float *)malloc(t.size * sizeof(float));
    if (t.data == NULL)
    {
        fprintf(stderr, "Erreur d'allocation de mémoire pour les données du tenseur.\n");
        exit(EXIT_FAILURE);
    }
    return t;
}

void free_tensor(Tensor *t)
{
    free(t->shape);
    free(t->data);
    t->data = NULL;
    t->shape = NULL;
    t->ndim = 0;
    t->size = 0;
}

Filter *initialize_filter(size_t filter_size, size_t input_channels, size_t output_channels)
{
    Filter *f = (Filter *)malloc(output_channels * sizeof(Filter));

    for (size_t i = 0; i < output_channels; i++)
    {
        f[i].data = (float ***)malloc(filter_size * sizeof(float **));
        for (size_t j = 0; j < filter_size; j++)
        {
            f[i].data[j] = (float **)malloc(filter_size * sizeof(float *));
            for (size_t k = 0; k < filter_size; k++)
            {
                f[i].data[j][k] = (float *)malloc(input_channels * sizeof(float));
            }
        }
    }

    return f;
}

void free_filter(Filter *f)
{
    for (size_t i = 0; i < f->output_channels; i++)
    {
        for (size_t j = 0; j < f->filter_size; j++)
        {
            for (size_t k = 0; k < f->filter_size; k++)
            {
                free(f[i].data[j][k]);
            }
            free(f[i].data[j]);
        }
        free(f[i].data);
    }
    free(f);
}


Layer dense_layer(size_t input_dim, size_t output_dim, void (*activation)(Tensor *))
{

    Layer l;
    l.type = DENSE;
    l.dense.input_dim = input_dim;
    l.dense.output_dim = output_dim;
    l.dense.activation = activation;

    size_t biases_shape[] = {output_dim};
    l.dense.biases = initialize_tensor(biases_shape, 1);

    size_t weights_shape[] = {input_dim, output_dim};
    l.dense.weights = initialize_tensor(weights_shape, 2);

    size_t output_data_shape[] = {output_dim};
    l.dense.output_data = initialize_tensor(output_data_shape, 1);
    // Initialisation des poids et des biais

    // Initialiser le générateur de nombres aléatoires
    srand((unsigned int)time(NULL));

    // Pour aller plus loin, modifier l'initialisation en fonction de la fonction d'activation
    for (size_t i = 0; i < output_dim * input_dim; i++)
    {
        l.dense.weights.data[i] = (float)rand() / RAND_MAX;
    }

    for (int i = 0; i < output_dim; i++)
    {
        l.dense.biases.data[i] = 0.0f;
    }
    return l;
}


Layer conv_layer(size_t input_height, size_t input_width, size_t filter_size, size_t input_channels, size_t output_channels, size_t stride, size_t padding, void (*activation)(Tensor *))
{
    Layer l;

    l.type = CONV;
    l.conv.input_height = input_height;
    l.conv.input_width = input_width;
    l.conv.input_channels = input_channels;
    l.conv.output_channels = output_channels;
    l.conv.stride = stride;
    l.conv.padding = padding;
    l.conv.activation = activation;

    // Initialize filters
    l.conv.filters = initialize_filter(filter_size, input_channels, output_channels);

    srand((unsigned int)time(NULL)); // Initialize the random number generator

    for (size_t i = 0; i < output_channels; i++)
    {
        for (size_t j = 0; j < filter_size; j++)
        {
            for (size_t k = 0; k < filter_size; k++)
            {
                for (size_t l = 0; l < input_channels; l++)
                {
                    l.conv.filters[i].data[j][k][l] = (float)rand() / RAND_MAX;
                }
            }
        }
    }

    // Biases are one per output channel
    size_t biases_shape[] = {output_channels};
    l.conv.biases = initialize_tensor(biases_shape, 1);

    // Initialize biases to zero
    for (int i = 0; i < output_channels; i++)
    {
        l.conv.biases.data[i] = 0.0f;
    }

    // Determine the size of the output data
    size_t output_height = (input_height - filter_size + 2 * padding) / stride + 1;
    size_t output_width = (input_width - filter_size + 2 * padding) / stride + 1;
    size_t output_data_shape[] = {output_height, output_width, output_channels};
    l.conv.output_data = initialize_tensor(output_data_shape, 3);

    return l;
}

void free_dense_layer(DenseLayer *l)
{
    free_tensor(&l->weights);
    free_tensor(&l->biases);
    free_tensor(&l->output_data);
    l->output_dim = 0;
    l->input_dim = 0;
    l->activation = NULL;
}

void free_conv_layer(ConvLayer *l)
{
    free_tensor(&l->biases);
    free_tensor(&l->output_data);
    free_filter(&l->filters);
    l->input_channels = 0;
    l->output_channels = 0;
    l->stride = 0;
    l->padding = 0;
    l->activation = NULL;
}

void free_layer(Layer *l)
{
    switch (l->type)
    {
    case DENSE:
        free_dense_layer(&l->dense);
        break;
    case CONV:
        free_conv_layer(&l->conv);
        break;
    default:
        fprintf(stderr, "Type de couche non supporté.\n");
        exit(EXIT_FAILURE);
    }
}

NeuralNetwork initialize_NeuralNetwork(Layer *layers, size_t num_layers)
{
    NeuralNetwork nn;

    nn.num_layers = num_layers;
    nn.layers = (Layer *)malloc(num_layers * sizeof(Layer));

    if (nn.layers == NULL)
    {
        fprintf(stderr, "Erreur d'allocation de mémoire pour le réseau de neurones");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_layers; i++)
    {
        nn.layers[i] = layers[i];
    }
    return nn;
}

void free_NeuralNetwork(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->num_layers; i++)
    {
        free_layer(&nn->layers[i]);
    }
    free(nn->layers);
    nn->layers = NULL;
    nn->num_layers = 0;
}

void forward_pass(Layer *layer, Tensor *input, Tensor *output)
{
    switch (layer->type)
    {
    case DENSE:
        for (size_t i = 0; i < layer->dense.output_dim; i++)
        {
            layer->dense.output_data.data[i] = layer->dense.biases.data[i];
            for (size_t j = 0; j < layer->dense.input_dim; j++)
            {
                layer->dense.output_data.data[i] += layer->dense.weights.data[i * layer->dense.input_dim + j] * input->data[j];
            }
        }
        layer->dense.activation(output);
        break;
    case CONV:
        size_t output_height = (layer->conv.input_height - layer->conv.filters->filter_size + 2 * layer->conv.padding) / layer->conv.stride;
        size_t output_width = (layer->conv.input_width - layer->conv.filters->filter_size + 2 * layer->conv.padding) / layer->conv.stride;

        if (layer->conv.output_data.shape[0] != output_width || layer->conv.output_data.shape[1] != output_height)
        {
            printf("Erreur: Les dimensions de la sortie ne correspondent pas à la forme de l'output_data.\n");
            return;
        }

        // Appliquer chaque filtre à l'entrée
        for (size_t oc = 0; oc < layer->conv.output_channels; oc++) {
            for (size_t oh = 0; oh < output_height; oh++) {
                for (size_t ow = 0; ow < output_width; ow++) {
                    float sum = 0.0f;

                    // Convolution 2D
                    for (size_t ic = 0; ic < layer->conv.input_channels; ic++) {
                        for (size_t fh = 0; fh < layer->conv.filters->filter_size; fh++) {
                            for (size_t fw = 0; fw < layer->conv.filters->filter_size; fw++) {
                                size_t ih = oh * layer->conv.stride + fh - layer->conv.padding;
                                size_t iw = ow * layer->conv.stride + fw - layer->conv.padding;

                                // Vérifier si l'indice est dans les limites de l'entrée (gérer le padding)
                                if (ih >= 0 && ih < layer->conv.input_height && iw >= 0 && iw < layer->conv.input_width) {
                                    sum += layer->conv.filters[oc].data[fh][fw][ic] * input->data[ih * layer->conv.input_width * layer->conv.input_channels + iw * layer->conv.input_channels + ic];
                                }
                            }
                        }
                    }
                    // Ajoutez le biais et appliquez l'activation
                    sum += layer->conv.biases.data[oc];
                    layer->conv.output_data.data[oh * output_width * layer->conv.output_channels + ow * layer->conv.output_channels + oc] = sum;
                }
            }
        }
        // Appliquer la fonction d'activation
        if (layer->conv.activation != NULL)
        {
            layer->conv.activation(&(layer->conv.output_data));
        }

        break;
    default:
        fprintf(stderr, "Type de couche non supporté.\n");
        exit(EXIT_FAILURE);
    }
}

void forward_pass_network(NeuralNetwork *nn, Tensor *input, Tensor *output)
{

    Tensor current_input = *input;
    Tensor current_output;

    for (size_t i = 0; i < nn->num_layers; i++)
    {

        switch (nn->layers[i].type)
        {
        case DENSE:
            size_t output_shape[] = {nn->layers[i].dense.output_dim};
            current_output = initialize_tensor(output_shape, 1);
            break;
        case CONV :
            size_t output_height = (nn->layers[i].conv.input_height - nn->layers[i].conv.filters->filter_size + 2 * nn->layers[i].conv.padding) / nn->layers[i].conv.stride + 1;
            size_t output_width = (nn->layers[i].conv.input_width - nn->layers[i].conv.filters->filter_size + 2 * nn->layers[i].conv.padding) / nn->layers[i].conv.stride + 1;
            size_t output_data_shape[] = {output_height, output_width, nn->layers[i].conv.output_channels};
            current_output = initialize_tensor(output_data_shape, 3);

        default:
            fprintf(stderr, "La couche est mal définie.\n");
            break;
        }

        forward_pass(&nn->layers[i], &current_input, &current_output);

        if (i > 0)
        {
            free_tensor(&current_input);
        }

        current_input = current_output;
    }

    *output = current_output;
}

void relu(Tensor *t)
{
    for (int i = 0; i < t->size; i++)
    {
        if (t->data[i] < 0)
        {
            t->data[i] = 0;
        }
    }
}

void sigmoid(Tensor *t)
{
    for (size_t i = 0; i < t->size; ++i)
    {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
}

Tensor gradient_activation(void (*activation)(Tensor *), Tensor *t)
{
    Tensor grad = initialize_tensor(t->shape, t->ndim);

    // Calcul du gradient en fonction de l'activation
    if (activation == relu)
    {
        for (size_t i = 0; i < t->size; i++)
        {
            grad.data[i] = (t->data[i] > 0) ? 1.0f : 0.0f;
        }
    }
    else if (activation == sigmoid)
    {
        for (size_t i = 0; i < t->size; ++i)
        {
            float sigmoid_val = 1.0f / (1.0f + expf(-t->data[i]));
            grad.data[i] = sigmoid_val * (1.0f - sigmoid_val);
        }
    }
    else
    {
        fprintf(stderr, "Fonction d'activation non supportée.\n");
        exit(EXIT_FAILURE);
    }

    return grad;
}

#define NUM_LAYERS 2

int main()
{
    Layer layers[NUM_LAYERS]; 
    layers[0] = dense_layer(2, 150, relu);
    layers[1] = dense_layer(150, 200, relu);

    NeuralNetwork nn = initialize_NeuralNetwork(layers, NUM_LAYERS);

    size_t input_shape[] = {2};
    Tensor input = initialize_tensor(input_shape, 1);

    input.data[0] = 1.0f;
    input.data[1] = 2.0f;

    Tensor output;
    forward_pass_network(&nn, &input, &output);

    printf("Output of the neural network:\n");
    for (size_t i = 0; i < output.size; i++)
    {
        printf("%f\n", output.data[i]);
    }
    printf("Output.size : %ld", output.size);

    free_tensor(&input);
    free_tensor(&output);
    free_NeuralNetwork(&nn);

    return 0;
}
