/**
 * \file basic/express/aifes_express_f32_fnn.h
 * \internal
 * \date 04.08.2021
 * \endinternal
 * \version 2.2.0
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
 * \brief AIfES Express functions for weights with F32 (float32) data type
 *
 * AIfES Express is a beginner friendly high-level API of AIfES.
 * This file contains all necessary functions for neural networks with float32 weights.
 */

#ifndef EXPRESS_FNN_F32
#define EXPRESS_FNN_F32

#include "core/aifes_core.h"
#include "core/aifes_math.h"

/** @brief Possible activation functions in AIfES-Express
 */
typedef enum
{
    AIfES_E_relu,           /**< Relu */
    AIfES_E_sigmoid,        /**< Sigmoid */
    AIfES_E_softmax,        /**< Softmax */
    AIfES_E_leaky_relu,     /**< Leaky_relu */
    AIfES_E_elu,            /**< Elu */
    AIfES_E_tanh,           /**< Tanh */
    AIfES_E_softsign,       /**< Softsign */
    AIfES_E_linear          /**< Linear */

}AIFES_E_activations;

/** @brief Possible loss functions in AIfES-Express
 */
typedef enum
{
    AIfES_E_mse,            /**< Mean squared error (MSE) */
    AIfES_E_crossentropy    /**< Crossentropy*/

}AIFES_E_loss;

/** @brief Possible optimizers in AIfES-Express
 */
typedef enum
{
    AIfES_E_adam,           /**< ADAM */
    AIfES_E_sgd             /**< SGD */

}AIFES_E_optimizer;

/** @brief Possible weight initialization methods in AIfES-Express
 */
typedef enum
{
    AIfES_E_init_uniform,           /**< Dices the weights in a range of values you specify */
    AIfES_E_init_glorot_uniform,    /**< Random numbers are uniformly diced within a certain range */
    AIfES_E_init_no_init            /**< No weight init. If starting weights are already available or if you want to continue training */

}AIFES_E_init_weights_method;

/** @brief Turn early stopping on or off
 */
typedef enum
{
    AIfES_E_early_stopping_off,
    AIfES_E_early_stopping_on

}AIFES_E_early_stopping;

/** @brief Parameters for the FNN model
 *
 * This structure contains all parameters for the creation of the model.
 *
 * **Example:**\n
 * In this example, a 2-3-1 FNN is generated.
 * The hidden layer and the output layer use a sigmoid activation function
 * \code{.c}
 * uint32_t nn_structure[3] = {2,3,1};
 *
 * AIFES_E_activations nn_activations[2];
 * nn_activations[0] = AIfES_E_sigmoid;
 * nn_activations[1] = AIfES_E_sigmoid;
 *
 * float weights[] = {-10.1164f, -8.4212f, 5.4396f, 7.297f, -7.6482f, -9.0155f, -2.9653f,
 *                     2.3677f, -1.5968f, 12.0305f, -6.5858f, 11.9371f,-5.4247f};
 *
 * AIFES_E_model_parameter_fnn_f32 nn;
 * nn.layer_count = 3;
 * nn.fnn_structure = nn_structure;
 * nn.fnn_activations = nn_activations;
 * nn.flat_weights = weights;
 * \endcode
 */
struct AIFES_E_model_parameter_fnn_f32{
   uint32_t             layer_count;        /**< Count of all layers, including input and output layers */
   uint32_t             *fnn_structure;     /**< Pointer to the network structure */
   AIFES_E_activations  *fnn_activations;   /**< Pointer to the activation function list (AIFES_E_activations) */
   void *               flat_weights;       /**< Pointer to the array with the weights */
};

/** @brief Parameters for Training
 *
 * The structure includes all parameters for training.
 *
 * **Example:**
 * \code{.c}
 * AIFES_E_training_parameter_fnn_f32  nn_train_config;
 * nn_train_config.optimizer = AIfES_E_adam;
 * nn_train_config.sgd_momentum = 0.0f;         // Only for SGD optimizer
 * nn_train_config.loss = AIfES_E_mse;
 * nn_train_config.learn_rate = 0.05f;
 * nn_train_config.batch_size = 4;
 * nn_train_config.epochs = 100;
 * nn_train_config.epochs_loss_print_interval = 10;
 *
 * // Your individual print function
 * nn_train_config.loss_print_function = my_print_function;
 *
 * //You can enable early stopping, so that learning is automatically stopped when a learning target is reached
 * nn_train_config.early_stopping = AIfES_E_early_stopping_on;
 * //Define your target loss
 * nn_train_config.early_stopping_target_loss = 0.004;
 * \endcode
 */
struct AIFES_E_training_parameter_fnn_f32{
   AIFES_E_loss                 loss;   /**< Loss selection (AIFES_E_loss) */
   AIFES_E_optimizer            optimizer;  /**< Optimizer selection (AIFES_E_optimizer) */
   float                        learn_rate; /**< Learning rate for training (For all optimizers) */
   float                        sgd_momentum; /**< Optional momentum for SGD (Value 0.0f means Momentum off) */
   uint32_t                     batch_size; /**< Batch size (min: 1 -> max: Entire data set) */
   uint32_t                     epochs; /**< Number of desired epochs (If early stopping is on, can be stopped before) */
   uint32_t                     epochs_loss_print_interval; /**< Selection of the interval in which the loss is to be calculated and output via the print function */

    /** @brief Individual print function for the loss
    *
    * The loss is calculated only in the previously selected interval and output via the function to which the pointer points
    * The function must have the following structure
    * \code{.c}
    * void my_print_function(float loss)
    * {
    *    //E.g. a normal print output
    *    printf("Loss: %f\n", loss);
    * }
    * \endcode
    *
    * @param tensor           The tensor to print
    */
   void                         (*loss_print_function)(float);
   AIFES_E_early_stopping       early_stopping; /**< Switch early stopping on or off */
   float                        early_stopping_target_loss; /**< If early stopping is switched on, the target loss can be specified here. It is only checked in the print interval */
};

typedef struct AIFES_E_model_parameter_fnn_f32 	AIFES_E_model_parameter_fnn_f32;

typedef struct AIFES_E_training_parameter_fnn_f32 	AIFES_E_training_parameter_fnn_f32;

typedef struct AIFES_E_init_weights_parameter_fnn_f32 	AIFES_E_init_weights_parameter_fnn_f32;


/** @brief Calculates the total required float weights for the selected network structure
 *
 *
 * @param       *fnn_structure      The FNN structure
 * @param       layer_count         Number of layers
 * @return      Number of float weights required
 */
uint32_t AIFES_E_flat_weights_number_fnn_f32(uint32_t *fnn_structure,uint32_t layer_count);


/** @brief Creates and compiles an FNN model
 *
 * Creates the layers defined by the FNN model parameters, compiles the model and distributes the model weights.
 *
 * Possible returns:
 * * 0 = success
 * * -1 = ERROR! Unknown activation function
 * * -2 = ERROR! Unknown loss function
 * * -3 = ERROR! Not enough memory
 * * -4 = ERROR! Use the crossentropy as loss for softmax
 *
 * This function must be called before setting/using the model for inference or training.
 *
 * @param       *AIFES_E_fnn                The FNN model parameters
 * @param       *AIFES_E_fnn_training       The training parameters (can be NULL if no training is required)
 * @param       *model                      The model to configure
 * @param       **model_memory              Pointer to the model memory
 * @return      Error output
 */
int8_t AIFES_E_create_fnn_f32(AIFES_E_model_parameter_fnn_f32 *AIFES_E_fnn, AIFES_E_training_parameter_fnn_f32 *AIFES_E_fnn_training, aimodel_t *model, void **model_memory);


/** @brief Frees the layer memory used to store an FNN model
 *
 * Releases the memory allocated for the layers of the model by AIFES_E_create_fnn_f32().
 *
 * @param       **model_memory      Pointer to the model memory
 */
void AIFES_E_free_model_memory_fnn_f32(void **model_memory);



/** @brief Sets an FNN model in inference mode
 *
 * Calculates the memory required for intermediate inference results, allocates the memory and schedules it over the
 * compiled model. The configured memory can be reused for multiple calls to AIFES_E_inference_fnn_f32().
 *
 * Possible returns:
 * * 0 = success
 * * -1 = ERROR! Not enough memory
 * 
 * This function must be called once before using the model for inference. The model must be created with
 * AIFES_E_create_fnn_f32() before calling this function.
 *
 * @param       *model             The compiled FNN model
 * @param       **inference_memory Pointer used for inference memory configuration
 * @return      Error output
 */
int8_t AIFES_E_set_inference_mode_fnn_f32(aimodel_t *model, void **inference_memory);


/** @brief Performs inference using a pre-configured FNN model
 * 
 * Performs inference using a pre-configured model created with AIFES_E_create_fnn_f32() and inference memory configured
 * with AIFES_E_set_inference_mode_fnn_f32(). The same model and inference memory can be reused for multiple inferences.
 * 
 * Possible returns:
 * * 0 = success
 * * -1 = ERROR! Tensor dtype
 * * -2 = ERROR! Tensor shape: Data Number
 * * -3 = ERROR! Input tensor shape does not correspond to ANN inputs
 * * -4 = ERROR! Output tensor shape does not correspond to ANN outputs
 *
 * @param       *input_tensor      Tensor with the inputs
 * @param       *model             The configured FNN model
 * @param       *output_tensor     Tensor for the results
 * @return      Error output
 */
int8_t AIFES_E_inference_fnn_f32(aitensor_t *input_tensor, aimodel_t *model, aitensor_t *output_tensor);


/** @brief Frees the memory used to run inference with an FNN model
 *
 * Releases the memory allocated for inference by AIFES_E_set_inference_mode_fnn_f32().
 *
 * @param       *inference_memory  Pointer to the inference memory
 */
void AIFES_E_free_inference_memory_fnn_f32(void **inference_memory);


/** @brief Sets an FNN model in training mode
 *
 * Calculates the memory required for intermediate training results, allocates the memory and schedules it over the
 * compiled model. The configured memory can be reused for multiple calls to AIFES_E_training_fnn_f32().
 *
 * Possible returns:
 * * 0 = success
 * * -1 = ERROR! Model has no loss function configured for training
 * * -2 = ERROR! learn_rate or sgd_momentum negative
 * * -3 = ERROR! Unknown optimizer
 * * -4 = ERROR! Not enough memory
 *
 * This function must be called once before using the model for training. The model must be created with
 * AIFES_E_create_fnn_f32() before calling this function.
 *
 * @param       *AIFES_E_fnn_training       The training parameters
 * @param       *model                      The compiled FNN model
 * @param       **optimizer                 Pointer used for optimizer configuration
 * @param       **training_memory           Pointer used for training memory configuration
 * @return      Error output
 */
int8_t AIFES_E_set_training_mode_fnn_f32(AIFES_E_training_parameter_fnn_f32 *AIFES_E_fnn_training, aimodel_t *model, aiopti_t **optimizer, void **training_memory);


/** @brief Performs training using a pre-configured FNN model
 *
 * Performs training using a pre-configured model created with AIFES_E_create_fnn_f32() and training memory configured
 * with AIFES_E_set_training_mode_fnn_f32(). The same model and training memory can be reused for multiple trainings.
 *
 * Possible returns:
 * * 0 = success
 * * -1 = ERROR! Tensor dtype
 * * -2 = ERROR! Tensor shape: Data Number
 * * -3 = ERROR! Input tensor shape does not correspond to ANN inputs
 * * -4 = ERROR! Output or target tensor shape does not correspond to ANN outputs\n
 * * -5 = ERROR! batch_size: min = 1 / max = Number of training data
 * * -6 = ERROR! Unknown loss function
 *
 * @param       *input_tensor               Tensor with the inputs
 * @param       *target_tensor              Tensor with the target outputs
 * @param       *AIFES_E_fnn_training       The training parameters
 * @param       *model                      The configured FNN model
 * @param       *optimizer                  Pointer used for optimizer configuration
 * @param       *output_tensor              Tensor for the results
 * @return      Error output
 */
int8_t AIFES_E_training_fnn_f32(aitensor_t *input_tensor,
                                aitensor_t *target_tensor,
                                AIFES_E_training_parameter_fnn_f32 *AIFES_E_fnn_training,
                                aimodel_t *model,
                                aiopti_t *optimizer,
                                aitensor_t *output_tensor);


/** @brief Frees the memory used to run training with an FNN model
 *
 * Releases the memory allocated for training by AIFES_E_set_training_mode_fnn_f32().
 *
 * @param       **optimizer         Pointer to the optimizer
 * @param       **training_memory   Pointer to the training memory
 */
void AIFES_E_free_training_memory_fnn_f32(aiopti_t **optimizer, void **training_memory);

#endif // EXPRESS_FNN_F32
