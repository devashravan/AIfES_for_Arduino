/**
 * \file basic/express/aifes_express_f32_fnn.c
 * \version 2.2.0
 * \date 04.08.2021
 * \copyright  Copyright (C) 2020-2023  Fraunhofer Institute for Microelectronic Circuits and Systems.
    All rights reserved.<br><br>
    AIfES is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.<br><br>
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.<br><br>
    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * \brief
 * \details
 */

#include "basic/express/aifes_express_f32_fnn.h"


// Include the layers in default implementation
#include "basic/default/ailayer/ailayer_dense_default.h"
#include "basic/default/ailayer/ailayer_input_default.h"
#include "basic/default/ailayer/ailayer_relu_default.h"
#include "basic/default/ailayer/ailayer_leaky_relu_default.h"
#include "basic/default/ailayer/ailayer_elu_default.h"
#include "basic/default/ailayer/ailayer_sigmoid_default.h"
#include "basic/default/ailayer/ailayer_tanh_default.h"
#include "basic/default/ailayer/ailayer_softmax_default.h"
#include "basic/default/ailayer/ailayer_softsign_default.h"

// Include the losses in default implementation
#include "basic/default/ailoss/ailoss_mse_default.h"
#include "basic/default/ailoss/ailoss_crossentropy_default.h"

// Include the optimizers in default implementation
#include "basic/default/aiopti/aiopti_sgd_default.h"
#include "basic/default/aiopti/aiopti_adam_default.h"

// Include the algorithmic
#include "basic/base/aialgo/aialgo_sequential_inference.h"
#include "basic/base/aialgo/aialgo_sequential_training.h"


uint32_t AIFES_E_flat_weights_number_fnn_f32(uint32_t *fnn_structure,uint32_t layer_count)
{
    uint32_t i = 0;
    uint32_t sum = 0;


    for (i = 1; i < layer_count; i++)
    {
        sum = sum + fnn_structure[i-1] * fnn_structure[i] + fnn_structure[i];
    }

    return sum;
}


int8_t AIFES_E_create_fnn_f32(AIFES_E_model_parameter_fnn_f32 *AIFES_E_fnn, AIFES_E_training_parameter_fnn_f32 *AIFES_E_fnn_training, aimodel_t *model, void **model_memory)
{
    uint8_t for_training = (AIFES_E_fnn_training != NULL);

    uint32_t i = 0;

    // Counter for the activation functions
    uint32_t relu_count = 0;
    uint32_t sigmoid_count = 0;
    uint32_t softmax_count = 0;
    uint32_t leaky_relu_count = 0;
    uint32_t elu_count = 0;
    uint32_t tanh_count = 0;
    uint32_t softsign_count = 0;
    uint32_t linear_count = 0;

    // Count for the activation functions in the FNN structure
    for (i = 0; i < AIFES_E_fnn->layer_count - 1; i++) {
        switch(AIFES_E_fnn->fnn_activations[i]) {
            case AIfES_E_relu:
                relu_count ++;
                break;
            case AIfES_E_sigmoid:
                sigmoid_count ++;
                break;
            case AIfES_E_softmax:
                softmax_count ++;
                break;
            case AIfES_E_leaky_relu:
                leaky_relu_count ++;
                break;
            case AIfES_E_elu:
                elu_count ++;
                break;
            case AIfES_E_tanh:
                tanh_count ++;
                break;
            case AIfES_E_softsign:
                softsign_count ++;
                break;
            case AIfES_E_linear:
                linear_count ++;
                break;
            default :
                aiprint("ERROR! Unknown activation function\n");
                return(-1);
       }
    }

    uint32_t model_memory_size = relu_count * sizeof(ailayer_relu_f32_t) +
                                 sigmoid_count * sizeof(ailayer_sigmoid_f32_t) +
                                 softmax_count * sizeof(ailayer_softmax_f32_t) +
                                 leaky_relu_count * sizeof(ailayer_leaky_relu_f32_t) +
                                 elu_count * sizeof(ailayer_elu_f32_t) +
                                 tanh_count * sizeof(ailayer_tanh_f32_t) +
                                 softsign_count * sizeof(ailayer_softsign_f32_t) +
                                 (AIFES_E_fnn->layer_count - 1) * sizeof(ailayer_dense_t) +
                                 sizeof(ailayer_input_t) +
                                 2 * sizeof(uint16_t) +
                                 (for_training ? (sizeof(ailoss_mse_t) + sizeof(ailoss_crossentropy_t)) : 0);

    aiprint("Memory required to store the model: ");
    aiprint_uint("%u", model_memory_size);
    aiprint(" bytes\n");

    *model_memory = calloc(1, model_memory_size);

    if(*model_memory == NULL)
    {
        aiprint("ERROR! Not enough memory\n");
        return(-3);
    }

    // Distribute the memory
    uint8_t *p = *model_memory;

    ailayer_relu_f32_t *relu_layer = (ailayer_relu_f32_t *)p;                     p += relu_count * sizeof(ailayer_relu_f32_t);
    ailayer_sigmoid_f32_t *sigmoid_layer = (ailayer_sigmoid_f32_t *)p;            p += sigmoid_count * sizeof(ailayer_sigmoid_f32_t);
    ailayer_softmax_f32_t *softmax_layer = (ailayer_softmax_f32_t *)p;            p += softmax_count * sizeof(ailayer_softmax_f32_t);
    ailayer_leaky_relu_f32_t *leaky_relu_layer = (ailayer_leaky_relu_f32_t *)p;   p += leaky_relu_count * sizeof(ailayer_leaky_relu_f32_t);
    ailayer_elu_f32_t *elu_layer = (ailayer_elu_f32_t *)p;                        p += elu_count * sizeof(ailayer_elu_f32_t);
    ailayer_tanh_f32_t *tanh_layer = (ailayer_tanh_f32_t *)p;                     p += tanh_count * sizeof(ailayer_tanh_f32_t);
    ailayer_softsign_f32_t *softsign_layer = (ailayer_softsign_f32_t *)p;         p += softsign_count * sizeof(ailayer_softsign_f32_t);
    ailayer_dense_t *dense_layer = (ailayer_dense_t *)p;                          p += (AIFES_E_fnn->layer_count - 1) * sizeof(ailayer_dense_t);
    ailayer_input_t *input_layer = (ailayer_input_t *)p;                          p += sizeof(ailayer_input_t);
    ailoss_mse_t *mse_loss = NULL;
    ailoss_crossentropy_t *crossentropy_loss = NULL;

    if (for_training) {
        mse_loss = (ailoss_mse_t *)p;                                             p += sizeof(ailoss_mse_t);
        crossentropy_loss = (ailoss_crossentropy_t *)p;                           p += sizeof(ailoss_crossentropy_t);
    }

    uint16_t *input_layer_shape = (uint16_t *)p;                                  p += 2 * sizeof(uint16_t);

    // Init leaky relu with default alpha
    for (i = 0; i < leaky_relu_count; i++) {
     leaky_relu_layer[i].alpha = 0.01f;
    }
    // Init elu with default alpha
    for (i = 0; i < elu_count; i++) {
     elu_layer[i].alpha = 1.0f;
    }

    // Use the variables as counter
    relu_count = 0;
    sigmoid_count = 0;
    softmax_count = 0;
    leaky_relu_count = 0;
    elu_count = 0;
    tanh_count = 0;
    softsign_count = 0;
    linear_count = 0;


    // Input layer
    input_layer_shape[0] = 1;
    input_layer_shape[1] = AIFES_E_fnn->fnn_structure[0];
    input_layer->input_dim = 2;                    // Definition of the input dimension (Must fit to the input tensor)
    input_layer->input_shape = input_layer_shape;  // Handover of the input layer shape

    ailayer_t *x;     // Layer object from AIfES, contains the layers

    // Generate the dense layer and the output layer
    for (i = 0; i < AIFES_E_fnn->layer_count - 1; i++) {

        if(i==0)
        {
            // Input layer
            model->input_layer = ailayer_input_f32_default(input_layer);
            // first dense layer
            dense_layer[i].neurons =  AIFES_E_fnn->fnn_structure[i+1];
            // Create first dense layer
            x = ailayer_dense_f32_default(&dense_layer[i], model->input_layer);
        }
        else
        {
            dense_layer[i].neurons =  AIFES_E_fnn->fnn_structure[i+1];  // Don't use the input neurons
            x = ailayer_dense_f32_default(&dense_layer[i], x);
        }

        switch(AIFES_E_fnn->fnn_activations[i]) {
            case AIfES_E_relu:
                if(i == AIFES_E_fnn->layer_count - 2){
                    model->output_layer = ailayer_relu_f32_default(&relu_layer[relu_count], x);  // Output layer
                }else{
                    x = ailayer_relu_f32_default(&relu_layer[relu_count], x);
                }
                relu_count ++;
                break;
            case AIfES_E_sigmoid:
                if(i == AIFES_E_fnn->layer_count - 2){
                    model->output_layer = ailayer_sigmoid_f32_default(&sigmoid_layer[sigmoid_count], x);
                }else{
                    x = ailayer_sigmoid_f32_default(&sigmoid_layer[sigmoid_count], x);
                }
                sigmoid_count ++;
                break;
            case AIfES_E_softmax:
                if(i == AIFES_E_fnn->layer_count - 2){
                    model->output_layer = ailayer_softmax_f32_default(&softmax_layer[softmax_count], x);
                }else{
                    x = ailayer_softmax_f32_default(&softmax_layer[softmax_count], x);
                }
                softmax_count ++;
                break;
            case AIfES_E_leaky_relu:
                if(i == AIFES_E_fnn->layer_count - 2){
                    model->output_layer = ailayer_leaky_relu_f32_default(&leaky_relu_layer[leaky_relu_count], x);
                }else{
                    x = ailayer_leaky_relu_f32_default(&leaky_relu_layer[leaky_relu_count], x);
                }
                leaky_relu_count ++;
                break;
            case AIfES_E_elu:
                if(i == AIFES_E_fnn->layer_count - 2){
                    model->output_layer = ailayer_elu_f32_default(&elu_layer[elu_count], x);
                }else{
                    x = ailayer_elu_f32_default(&elu_layer[elu_count], x);
                }
                elu_count ++;
                break;
            case AIfES_E_tanh:
                if(i == AIFES_E_fnn->layer_count - 2){
                    model->output_layer = ailayer_tanh_f32_default(&tanh_layer[tanh_count], x);
                }else{
                    x = ailayer_tanh_f32_default(&tanh_layer[tanh_count], x);
                }
                tanh_count ++;
                break;
            case AIfES_E_softsign:
                if(i == AIFES_E_fnn->layer_count - 2){
                    model->output_layer = ailayer_softsign_f32_default(&softsign_layer[softsign_count], x);
                }else{
                    x = ailayer_softsign_f32_default(&softsign_layer[softsign_count], x);
                }
                softsign_count ++;
                break;
            case AIfES_E_linear:
                if(i == AIFES_E_fnn->layer_count - 2){
                    model->output_layer = x;
                }else{
                    //no activation needed
                }
                linear_count ++;
                break;

            default :
                aiprint("ERROR! Unknown activation function\n");
                AIFES_E_free_model_memory_fnn_f32(model_memory);
                return(-1);
        }

    }

    model->loss = NULL;

    if (for_training) {
        switch(AIFES_E_fnn_training->loss) {
            case AIfES_E_mse:
                if (AIFES_E_fnn->fnn_activations[AIFES_E_fnn->layer_count - 2] == AIfES_E_softmax)
                {
                    aiprint("ERROR! Use the crossentropy as loss for softmax\n");
                    AIFES_E_free_model_memory_fnn_f32(model_memory);
                    return(-4);
                }
                model->loss = ailoss_mse_f32_default(mse_loss, model->output_layer);
                break;
            case AIfES_E_crossentropy:
                model->loss = ailoss_crossentropy_f32_default(crossentropy_loss, model->output_layer);
                break;
            default :
                aiprint("ERROR! Unknown loss function\n");
                AIFES_E_free_model_memory_fnn_f32(model_memory);
                return(-2);
        }
    }

    aialgo_compile_model(model); // Compile the AIfES model

    //Calculate required memory for the weights in Byte
    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(model);

    // Distribute the weights over the model
    aialgo_distribute_parameter_memory(model, (void *) AIFES_E_fnn->flat_weights, parameter_memory_size);
    return (0);
}


void AIFES_E_free_model_memory_fnn_f32(void **model_memory)
{
    free(*model_memory);
    *model_memory = NULL;
}


int8_t AIFES_E_set_inference_mode_fnn_f32(aimodel_t *model, void **inference_memory)
{
    // Allocate memory for result and temporal data
    uint32_t inference_memory_size = aialgo_sizeof_inference_memory(model);

    aiprint("Memory required to run inference: ");
    aiprint_uint("%u", inference_memory_size);
    aiprint(" bytes\n");

    *inference_memory = malloc(inference_memory_size);

    if(*inference_memory == NULL)
    {
        aiprint("ERROR! Not enough memory\n");
        return(-1);
    }

    // Schedule the memory over the model
    aialgo_schedule_inference_memory(model, *inference_memory, inference_memory_size);
    return (0);
}


int8_t AIFES_E_inference_fnn_f32(aitensor_t *input_tensor, aimodel_t *model, aitensor_t *output_tensor)
{
    if(input_tensor->dtype != aif32 || output_tensor->dtype != aif32)
    {
        aiprint("ERROR! Tensor dtype\n");
        return(-1);
    }

    if(input_tensor->shape[0] != output_tensor->shape[0])
    {
        aiprint("ERROR! Tensor shape: Data Number\n");
        return(-2);
    }

    if(input_tensor->shape[1] != model->input_layer->result.shape[1])
    {
        aiprint("ERROR! Input tensor shape does not correspond to ANN inputs\n");
        return(-3);
    }

    if(output_tensor->shape[1] != model->output_layer->result.shape[1])
    {
        aiprint("ERROR! Output tensor shape does not correspond to ANN outputs\n");
        return(-4);
    }

    //Do the inference and pass the result to the output_tensor
    aialgo_inference_model(model, input_tensor, output_tensor);
    return (0);
}


void AIFES_E_free_inference_memory_fnn_f32(void **inference_memory)
{
    free(*inference_memory);
    *inference_memory = NULL;
}


int8_t AIFES_E_set_training_mode_fnn_f32(AIFES_E_training_parameter_fnn_f32 *AIFES_E_fnn_training, aimodel_t *model, aiopti_t **optimizer, void **training_memory)
{
    if(model->loss == NULL)
    {
        aiprint("ERROR! Model has no loss function configured for training\n");
        return(-1);
    }

    if(AIFES_E_fnn_training->sgd_momentum < 0 || AIFES_E_fnn_training->learn_rate < 0)
    {
        aiprint("ERROR! learn_rate or sgd_momentum negative\n");
        return(-2);
    }
    
    *optimizer = NULL;
    uint32_t optimizer_memory_size = 0;

    switch(AIFES_E_fnn_training->optimizer) {
        case AIfES_E_adam:
            optimizer_memory_size = sizeof(aiopti_adam_f32_t);
            aiopti_adam_f32_t *adam_opti = malloc(optimizer_memory_size);

            if (adam_opti == NULL)
            {
                aiprint("ERROR! Not enough memory\n");
                return(-4);
            }

            adam_opti->learning_rate = AIFES_E_fnn_training->learn_rate;
            adam_opti->beta1 = 0.9f;
            adam_opti->beta2 = 0.999f;
            adam_opti->eps = 1e-7f;
            *optimizer = aiopti_adam_f32_default(adam_opti);
            break;
        case AIfES_E_sgd:
            optimizer_memory_size = sizeof(aiopti_sgd_f32_t);
            aiopti_sgd_f32_t *sgd_opti = malloc(optimizer_memory_size);

            if (sgd_opti == NULL)
            {
                aiprint("ERROR! Not enough memory\n");
                return(-4);
            }

            sgd_opti->learning_rate = AIFES_E_fnn_training->learn_rate;
            sgd_opti->momentum = AIFES_E_fnn_training->sgd_momentum;
            *optimizer = aiopti_sgd_f32_default(sgd_opti);
            break;
        default :
            aiprint("ERROR! Unknown optimizer\n");
            return(-3);
    }

    // Allocate memory for training and temporal data
    uint32_t training_memory_size = aialgo_sizeof_training_memory(model, *optimizer);

    aiprint("Memory required to run training: ");
    aiprint_uint("%u", training_memory_size + optimizer_memory_size);
    aiprint(" bytes\n");

    *training_memory = malloc(training_memory_size);

    if (*training_memory == NULL)
    {
        aiprint("ERROR! Not enough memory\n");
        free(*optimizer);
        *optimizer = NULL;
        return(-4);
    }

    // Schedule the memory over the model
    aialgo_schedule_training_memory(model, *optimizer, *training_memory, training_memory_size);
    
    // Initialize the AIfES model for training
    aialgo_init_model_for_training(model, *optimizer);
    return (0);
}


int8_t AIFES_E_training_fnn_f32(aitensor_t *input_tensor,
                                aitensor_t *target_tensor,
                                AIFES_E_training_parameter_fnn_f32 *AIFES_E_fnn_training,
                                aimodel_t *model,
                                aiopti_t *optimizer,
                                aitensor_t *output_tensor)
{
    uint32_t i = 0;

    if(input_tensor->dtype != aif32 || output_tensor->dtype != aif32 || target_tensor->dtype != aif32)
    {
        aiprint("ERROR! Tensor dtype\n");
        return(-1);
    }

    if(input_tensor->shape[0] != output_tensor->shape[0] || input_tensor->shape[0] != target_tensor->shape[0])
    {
        aiprint("ERROR! Tensor shape: Data Number\n");
        return(-2);
    }

    if(input_tensor->shape[1] != model->input_layer->result.shape[1])
    {
        aiprint("ERROR! Input tensor shape does not correspond to ANN inputs\n");
        return(-3);
    }

    if(output_tensor->shape[1] != model->output_layer->result.shape[1] || target_tensor->shape[1] != model->output_layer->result.shape[1])
    {
        aiprint("ERROR! Output or target tensor shape does not correspond to ANN outputs\n");
        return(-4);
    }

    if(AIFES_E_fnn_training->batch_size > input_tensor->shape[0] || AIFES_E_fnn_training->batch_size <= 0)
    {
        aiprint("ERROR! batch_size: min = 1 / max = Number of training data\n");
        return(-5);
    }

    float loss = 0;

    for(i = 0; i < AIFES_E_fnn_training->epochs; i++)
    {
        // One epoch of training. Iterates through the whole data once
        aialgo_train_model(model, input_tensor, target_tensor, optimizer, AIFES_E_fnn_training->batch_size);

        // Calculate and print loss every print_interval epochs
        if(i % AIFES_E_fnn_training->epochs_loss_print_interval == 0)
        {
            aialgo_calc_loss_model_f32(model, input_tensor, target_tensor, &loss);

            //Print the loss with the user function
            switch(AIFES_E_fnn_training->loss){
                case AIfES_E_mse:
                    // loss = loss / (NUMBER_OF_OUTPUT_NEURONS * NUMBER_DATASETS)
                    loss = loss / (model->output_layer->result.shape[1] * input_tensor->shape[0]);
                    (*AIFES_E_fnn_training->loss_print_function)(loss);
                    if(AIFES_E_fnn_training->early_stopping == AIfES_E_early_stopping_on)
                    {
                        if(loss <= AIFES_E_fnn_training->early_stopping_target_loss)
                        {
                            return(0);
                        }
                    }
                    break;
                case AIfES_E_crossentropy:
                    // loss = loss / NUMBER_DATASETS
                    loss = loss / input_tensor->shape[0];
                    (*AIFES_E_fnn_training->loss_print_function)(loss);

                    if(AIFES_E_fnn_training->early_stopping == AIfES_E_early_stopping_on)
                    {
                        if(loss <= AIFES_E_fnn_training->early_stopping_target_loss)
                        {
                            return(0);
                        }
                    }
                    break;
                default :
                    aiprint("ERROR! Unknown loss function\n");
                    return(-6);
            }
        }
    }

    // Do the inference after training
    //aialgo_inference_model(&model, input_tensor, output_tensor);

    return (0);
}


void AIFES_E_free_training_memory_fnn_f32(aiopti_t **optimizer, void **training_memory)
{
    free(*optimizer);
    *optimizer = NULL;

    free(*training_memory);
    *training_memory = NULL;
}
