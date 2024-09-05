#ifndef AI_H
#define AI_H


#define ALPHA 0.001       // Taux d'apprentissage
#define BETA1 0.9         // Décroissance du premier moment
#define BETA2 0.999       // Décroissance du deuxième moment
#define EPSILON 1e-8      // Valeur pour éviter la division par zéro

typedef struct {
    float* data;
    size_t *shape;
    size_t ndim;
    size_t size;
} Tensor;

typedef struct {
    Tensor weights;
    Tensor biases;
    Tensor output_data;
    size_t input_dim;
    size_t output_dim;
    void (*activation)(Tensor*);
} DenseLayer;

typedef struct {
    float ***data;  
    size_t filter_size;
    size_t input_channels;
    size_t output_channels;
} Filter;

typedef struct {
    Tensor biases;
    Tensor output_data;
    Filter *filters; 
    size_t input_channels;
    size_t output_channels;
    size_t input_height;
    size_t input_width;
    size_t stride;
    size_t padding;
    void (*activation)(Tensor*);
} ConvLayer;

typedef enum {
    DENSE,
    CONV
} LayerType;

typedef struct {
    LayerType type;
    union {
        DenseLayer dense;
        ConvLayer conv;
    };
} Layer;

typedef struct {
    size_t num_layers;
    Layer *layers;
} NeuralNetwork;



Tensor initialize_tensor (size_t *shape, size_t ndim);
void free_tensor (Tensor *t);
Layer dense_layer (size_t input_dim, size_t output_dim, void (*activation)(Tensor*));
void free_layer(Layer *l);
NeuralNetwork initialize_NeuralNetwork(Layer *layers, size_t num_layers);
void free_NeuralNetwork (NeuralNetwork *nn);
void forward_pass (Layer *layer, Tensor *input, Tensor *output);
void forward_pass_network (NeuralNetwork *nn, Tensor *input, Tensor *output);
void relu (Tensor *t);
void sigmoid(Tensor *t);
Tensor gradient_activation(void (*activation)(Tensor*), Tensor *t);


#endif // AI_H
