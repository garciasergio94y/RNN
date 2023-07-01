library(tfruns)
library(keras)

# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  flag_numeric("lookback", 10),
  flag_numeric("lstm_units", 32),
  flag_numeric("dropout1", 0.1),
  flag_numeric("recurrent_dropout1", 0.1),
  flag_numeric("L2regularizer1", 0.01),
  
  flag_numeric("dropout2", 0.1),
  flag_numeric("recurrent_dropout2", 0.1),
  flag_numeric("L2regularizer2", 0.01),
  
  flag_numeric("dense_u", 12),
  flag_string("optimizer", "adam"),
  flag_numeric("batch_size", 128)
)

# Generator functions------------------------------------------------------
# Con wavelets:
# train_data <- as.matrix(train_code[,-c(1,19)])
# Sin wavelets:
train_data <- as.matrix(train_code[,-c(1, 3:16, 19)])

lookback <- FLAGS$lookback
batch_size <- FLAGS$batch_size
delay <- 1
min_index <- 1
predseries <- 1

train_k <-  create_lstm_data.2(
  data = train_data,
  lookback = lookback,
  delay = delay,
  batch_size = batch_size,
  min_index = min_index,
  max_index = NULL,
  predseries = predseries)

# Cálculo del número de steps de training:
train_steps <- round((nrow(train_data)-lookback) / batch_size) 


wrap_train <- function() {
  seq <- NULL
  
  while (is.null(seq)) {
    tryCatch(
      seq <- train_k(),
      error = function(e) {
        warning("Empty sequence error occurred. Trying the next one.")
        return(NULL)  # Devuelve NULL si ocurre un error en la función
        # generadora y continúa el loop while.
      }
    )
  }
  
  tensor_list <- list(
    list(seq[[1]][, , 1, drop = FALSE],  # Sin wavelets
         #seq[[1]][, , 1:15, drop = FALSE], # Con wavelets
         seq[[1]][, , -1]),  # Sin wavelets
    #seq[[1]][, , 16:18]), # Con wavelets
    list(seq[[2]])
  )
  
  return(tensor_list)
}

#########################################################################
# Con wavelets:
# val_data <- as.matrix(val_code[,-c(1, 19)])
# Sin wavelets:
val_data <- as.matrix(val_code[,-c(1, 3:16, 19)])

val_k <- create_lstm_data.2(
  data = val_data,
  lookback = lookback,
  delay = delay,
  batch_size = batch_size,
  min_index = min_index,
  max_index = NULL,
  predseries = predseries)

# Cálculo del número de steps de training:
val_steps <- round((nrow(val_data)-lookback) / batch_size) 


wrap_val <- function() {
  seq <- NULL
  
  while (is.null(seq)) {
    tryCatch(
      seq <- val_k(),
      error = function(e) {
        warning("Empty sequence error occurred. Trying the next one.")
        return(NULL)  # Devuelve NULL si ocurre un error en la función
        # generadora y continúa el loop while.
      }
    )
  }
  
  tensor_list <- list(
    list(seq[[1]][, , 1, drop = FALSE],  # Sin wavelets
         #seq[[1]][, , 1:15, drop = FALSE], # Con wavelets
         seq[[1]][, , -1]),  # Sin wavelets
    #seq[[1]][, , 16:18]), # Con wavelets
    list(seq[[2]])
  )
  
  return(tensor_list)
}
##########################################################################
# Con wavelets:
# test_data <- as.matrix(test_code[,-c(1, 19)])
# Sin wavelets:
test_data <- as.matrix(test_code[,-c(1, 3:16, 19)])

test_k <-  create_lstm_data.2(
  data = test_data,
  lookback = lookback,
  delay = delay,
  batch_size = batch_size,
  min_index = min_index,
  max_index = NULL,
  predseries = predseries)

wrap_test <- function() {
  seq <- NULL
  
  while (is.null(seq)) {
    tryCatch(
      seq <- test_k(),
      error = function(e) {
        warning("Empty sequence error occurred. Trying the next one.")
        return(NULL)  # Devuelve NULL si ocurre un error en la función
        # generadora y continúa el loop while.
      }
    )
  }
  
  tensor_list <- list(
    list(seq[[1]][, , 1, drop = FALSE],  # Sin wavelets
         #seq[[1]][, , 1:15, drop = FALSE], # Con wavelets
         seq[[1]][, , -1]),  # Sin wavelets
    #seq[[1]][, , 16:18]), # Con wavelets
    list(seq[[2]])
  )
  
  return(tensor_list)
}

# Cálculo del número de steps de test:
test_steps <- round((nrow(test_data)-lookback) / batch_size) 

#######################################################################


# Model parameters--------------------------------------------------------
dynamic_features = 1 # Variables dinámicas (QC_RESULT)
# dynamic_features = 15 # Variables dinámicas (QC_RESULT + wavelets)
static_features = 3  # Variables estáticas (ANALIZADOR, CODIGO_PRUEBA Y NIVEL)
lstm_units = FLAGS$lstm_units # Número de unidades en la capa LSTM 
vocabulary_size = 7 # Número de niveles codificados de las variables estáticas.
epochs <- 20
optimizer <- "adam"
loss <- "mae"

# Define Model ------------------------------------------------------------

# Modelo LSTM:

# Capa de entrada de datos dinámicos:
dynamic_input_layer <- layer_input(shape = 
                                     c(lookback, dynamic_features), 
                                   name = "dynamic_input_layer")
#dynamic_input_layer <- layer_masking(mask_value =
#-99999)(dynamic_input_layer)
# Capa de entrada de datos estáticos: 
static_input_layer <- layer_input(shape = c(lookback,static_features), 
                                  name = "static_input_layer")

# Aplanado de la capa de datos estáticos antes del embedding:
flatten_static_layer <- layer_flatten()(static_input_layer)

# Capa a de embedding para los datos estáticos:
embedding_layer <- layer_embedding(
  input_dim = vocabulary_size,
  output_dim = lstm_units)(flatten_static_layer)

# Capa lambda para seleccionar los embeddings que modificarán los estados ocultos de la capa LSTM:
embedding_layer <- layer_lambda(f = \(x) x[,1,],
                                name = "lambda_embeddings")(embedding_layer)

# Capa LSTM que recibe dos valores de estado oculto de la capa de embeddings con la información de las variables estáticas:
lstm_layer_b1 <- layer_lstm(units = lstm_units,
                            name = "lstm_1", 
                            dropout = FLAGS$dropout1,
                            recurrent_dropout = FLAGS$recurrent_dropout1,
                            #kernel_regularizer = regularizer_l2(l=
                              #                      FLAGS$L2regularizer1),
                            return_sequences = F)(dynamic_input_layer,
                                                   initial_state = list(
                                                     embedding_layer,
                                                     embedding_layer))

#lstm_layer_b2 <- layer_lstm(units = lstm_units,
 #                           name = "lstm_2", 
  #                          dropout = FLAGS$dropout2,
   #                         recurrent_dropout = FLAGS$recurrent_dropout2,
    #                        kernel_regularizer = regularizer_l2(l=
     #                                               FLAGS$L2regularizer2),
      #                      return_sequences = F)(lstm_layer_b1)

dense1 <- layer_dense(units = FLAGS$dense_u, activation='relu')(lstm_layer_b1)


# Capa de salida con una unidad de activación lineal para la predicción:  
output_layer <- layer_dense(units = 1,
                            activation = "linear", 
                            name = "output_layer")(dense1)


# Create model-------------------------------------------------------------
model <- 
  keras_model(
    inputs  = list(dynamic_input_layer, static_input_layer),
    outputs = list(output_layer)
  )

# Compile-----------------------------------------------------------------
model %<>% compile(
  optimizer = optimizer,
  loss      = loss,
  metrics = list("mae",
                 "mse",
                 rmse
  )
)

# Save naïve model--------------------------------------------------------

# file_name <- paste0("lstm_QC_", 0, ".keras") 

save_model_weights_hdf5(model, 
                        file.path(resultsdir, "naive_model.h5"),
                        overwrite = T) 
# Callbacks---------------------------------------------------------------
current_time <- format(Sys.time(), "%Y%m%d%H%M%S")
file_name <- paste0("lstm_QC_op",current_time, ".keras")

callbacks <- list(
  callback_model_checkpoint(file.path(resultsdir, file_name),
                            monitor = "rmse",
                            mode = "min",
                            verbose=1,
                            save_best_only = T, 
                            save_weights_only = T,
                            ),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 0.000001),
  callback_early_stopping(monitor = "val_loss", patience = 20,
                          verbose = 1)
)

# Define metric rmse ------------------------------------------------------

K <- keras::backend()

rmse <- function(y_pred, y_true) {
  sq_diff <- K$square(y_pred - y_true)
  mean_sq_diff <- K$mean(sq_diff)
  rmse_value <- K$sqrt(mean_sq_diff)
  return(rmse_value)
}

attr(rmse, "py_function_name") <- "rmse"


# Train model--------------------------------------------------------------

history <- model %>%
  fit(wrap_train,
      epochs = epochs,
      steps_per_epoch = train_steps,
      validation_data = wrap_val,
      validation_steps = val_steps,
      callbacks = callbacks
  )

plot(history)

# Evaluate model-----------------------------------------------------------

evaluation_result <- evaluate(model, wrap_test, steps = test_steps)

evaluation_result

