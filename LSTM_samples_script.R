library(tfruns)
library(keras)

# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  flag_numeric("lookback", 20),
  flag_numeric("lstm_units", 32),
  flag_numeric("conv_filters", 24),
  flag_numeric("conv_kernels", 3),
  flag_numeric("dropout1", 0.3),
  flag_numeric("recurrent_dropout1", 0.3),
  flag_numeric("L2regularizer1", 0.01),
  
  flag_numeric("dropout2", 0.3),
  flag_numeric("recurrent_dropout2", 0.3),
  flag_numeric("L2regularizer2", 0.01),
  
  flag_numeric("dense_u", 24),
  
  flag_string("optimizer", "adam"),
  flag_numeric("batch_size", 256)
)

# Generator functions------------------------------------------------------
train_data <- as.matrix(train_code[,-1])
lookback <- FLAGS$lookback
batch_size <- FLAGS$batch_size
delay <- 0
min_index <- 0
predseries <- ncol(train_data)

train_k <-  create_lstm_data.1(
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
    list(seq[[1]][, , 1, drop = FALSE],
         seq[[1]][, , -1]),
    list(seq[[2]])
  )
  
  return(tensor_list)
}

#########################################################################

val_data <- as.matrix(val_code[,-1])

val_k <- create_lstm_data.1(
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
    list(seq[[1]][, , 1, drop = FALSE],
         seq[[1]][, , -1]),
    list(seq[[2]])
  )
  
  return(tensor_list)
}
##########################################################################

test_data <- as.matrix(test_code[,-1])

test_k <-  create_lstm_data.1(
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
    list(seq[[1]][, , 1, drop = FALSE],
         seq[[1]][, , -1]),
    list(seq[[2]])
  )
  
  return(tensor_list)
}

# Cálculo del número de steps de test:
test_steps <- round((nrow(test_data)-lookback) / batch_size) 

#######################################################################


# Model parameters--------------------------------------------------------
dynamic_features = 1 # Variables dinámicas (QC_RESULT)
static_features = 4  # Variables estáticas (ANALIZADOR, CODIGO_PRUEBA,
                      #EDAD_BIN Y PACIENTE_SEXO )
lstm_units = FLAGS$lstm_units # Número de unidades en la capa LSTM 
conv_filters = FLAGS$conv_filters # Número de filtros capa convolucional
conv_kernels = FLAGS$conv_kernels # Número de kernels
vocabulary_size = length(unique(train_code$PACIENTE_SEXO)) +
  length(unique(train_code$EDAD_BIN))+
  length(unique(train_code$ANALIZADOR))+
  length(unique(train_code$CODIGO_PRUEBA)) 
# Número de niveles codificados de las variables estáticas.

epochs <- 15

optimizer = optimizer_adam(learning_rate = 0.001)

loss = loss_sigmoid_focal_crossentropy(alpha = 0.8, 
                                       gamma = 4, 
                                       reduction = 
                                         tf$keras$losses$Reduction$AUTO)

# Define Model ------------------------------------------------------------

# Modelo LSTM:

# Capa de entrada de datos dinámicos:
dynamic_input_layer <- layer_input(
  shape = c(lookback, dynamic_features), 
  name = "dynamic_input_layer")

# Capa de entrada de datos estáticos:
static_input_layer <- layer_input(
  shape = c(lookback, static_features), 
  name = "static_input_layer")

# Aplanado de la capa de datos estáticos antes del embedding:
flatten_static_layer <- layer_flatten()(static_input_layer)

# Capa de embedding para los datos estáticos:
embedding_layer <- layer_embedding(
  input_dim = vocabulary_size,
  output_dim = lstm_units)(flatten_static_layer)

# Capa lambda para seleccionar los embeddings que modificarán los estados ocultos de la capa LSTM:
embedding_layer <- layer_lambda(
  f = \(x) x[,1,], name = "lambda_embeddings")(embedding_layer)

# Capa LSTM que recibe los valores de estado oculto de la capa convolución con las características que ésta haya extraído:
lstm_layer_1 <- layer_lstm(units = lstm_units,
                           name = "lstm_1", 
                           dropout = 0.3,
                           recurrent_dropout = 0.3,
                           return_sequences = T)(dynamic_input_layer,
                                                 initial_state = list(
                                                   embedding_layer,
                                                   embedding_layer))

# Capa convolucional:
conv_layer_2 <- layer_conv_1d(filters = conv_filters,
                            kernel_size = conv_kernels,
                            activation = "relu",
                            name = "conv_layer")(lstm_layer_1)


# Capa densa:
dense_layer <- layer_dense(units = 12,
                           activation = "relu",
                           name = "dense_layer")(conv_layer_2)

# Capa de salida con una unidad de activación lineal para la predicción:  
output_layer <- layer_dense(units = 1,
                            activation = "sigmoid",
                            name = "output_layer")(dense_layer)


# Create model-------------------------------------------------------------
model <- 
  keras_model(
    inputs  = list(dynamic_input_layer, static_input_layer),
    outputs = list(output_layer)
  )

# Funciones de métricas---------------------------------------------------
# Funciones para calcular precisión, recall y F1 como métricas para los
# modelos:

K <- backend()

precision <- function(y_true, y_pred) {
  # Verdaderos positivos: 
  true_positives <- sum(K$round(K$clip(y_true * y_pred, 0, 1)))
  
  # Posibles positivos: 
  possible_positives <- sum(K$round(K$clip(y_true, 0, 1)))
  
  # Positivos predichos:
  predicted_positives <- sum(K$round(K$clip(y_pred, 0, 1)))
  
  # Precisión: 
  precision <- true_positives / (predicted_positives + K$epsilon())
  
  return(precision)
}

recall <- function(y_true, y_pred) {
  # Verdaderos positivos: 
  true_positives <- sum(K$round(K$clip(y_true * y_pred, 0, 1)))
  
  # Posibles positivos: 
  possible_positives <- sum(K$round(K$clip(y_true, 0, 1)))
  
  # Recall:
  recall <- true_positives / (possible_positives + K$epsilon())
  
  return(recall)
}

f1_score <- function(y_true, y_pred) {
  # Verdaderos positivos: 
  true_positives <- sum(K$round(K$clip(y_true * y_pred, 0, 1)))
  
  # Posibles positivos: 
  possible_positives <- sum(K$round(K$clip(y_true, 0, 1)))
  
  # Positivos predichos:
  predicted_positives <- sum(K$round(K$clip(y_pred, 0, 1)))
  
  # Precisión: 
  precision <- true_positives / (predicted_positives + K$epsilon())
  
  # Recall:
  recall <- true_positives / (possible_positives + K$epsilon())
  
  # F1 score:
  f1_score <- 2 * precision * recall / (precision + recall +
                                          K$epsilon())
  return(f1_score)
}

# Asignación de los nombres de las funciones como atributos py_function_name:

attr(f1_score, "py_function_name") <- "f1_score"
attr(precision, "py_function_name") <- "precision"
attr(recall, "py_function_name") <- "recall"

# Compile-----------------------------------------------------------------
model %<>% compile(
  optimizer = optimizer,
  loss      = loss,
  metrics = list("binary_accuracy",
                 precision,
                 recall,
                 f1_score)
  )

# Save naïve model--------------------------------------------------------

file_name <- paste0("lstm_QC_", 0, ".keras") 

save_model_weights_hdf5(model, 
                        file.path(resultsdir, "naive_model.h5"),
                        overwrite = T) 
# Callbacks---------------------------------------------------------------

# Callbacks

callbacks <- list(
  callback_model_checkpoint(file.path(resultsdir,
                                       "LSTM_samples_101"),
                             monitor = "f1_score",
                             mode = "max",
                             verbose=1,
                             save_best_only = T, 
                             save_weights_only = T),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 0.000001),
  callback_early_stopping(monitor = "val_loss", patience = 20,
                          verbose = 1)
)




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

