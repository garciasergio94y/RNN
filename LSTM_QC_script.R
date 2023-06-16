library(tfruns)
library(keras)

# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  flag_numeric("lookback", 10),
  flag_numeric("units", 8),
  flag_numeric("dropout1", 0.1),
  flag_numeric("recurrent_dropout1", 0.1),
  flag_numeric("L2regularizer1", 0.01)
   

  #flag_numeric("units2", 8),
  #flag_numeric("dropout2", 0.1),
  #flag_numeric("recurrent_dropout2", 0.1),
  #flag_numeric("L2regularizer2", 0.01),
)

# Generator functions------------------------------------------------------
train_data <- as.matrix(train_code$data[,-1])
lookback <- FLAGS$lookback
batch_size <- 100
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

val_data <- as.matrix(val_code$data[,-1])

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

test_data <- as.matrix(test_code$data[,-1])

test_k <-  create_lstm_data.2(
  data = test_data,
  lookback = lookback,
  delay = delay,
  batch_size = batch_size,
  min_index = min_index,
  max_index = NULL,
  predseries = predseries)


# Cálculo del número de steps de training:
test_steps <- round((nrow(test_data)-lookback) / batch_size) 

# Define Model ------------------------------------------------------------

model_0 <- keras_model_sequential() %>%
  #layer_masking(mask_value = -99999, input_shape = c(NULL, k, p)) %>%
  layer_lstm(units = FLAGS$units, 
             dropout = FLAGS$dropout1, 
             recurrent_dropout = FLAGS$recurrent_dropout1,
             kernel_regularizer = regularizer_l2(l=FLAGS$L2regularizer1),
             return_sequences = F,
             input_shape = c(NULL, k, p)) %>%
  #layer_lstm(units = 64, 
  #          dropout=0.8, 
  #         recurrent_dropout=0.5) %>%
  layer_dense(units = 1,
              activation = "sigmoid")


# Define metric rmse ------------------------------------------------------

K <- keras::backend()

rmse <- function(y_pred, y_true) {
  sq_diff <- K$square(y_pred - y_true)
  mean_sq_diff <- K$mean(sq_diff)
  rmse_value <- K$sqrt(mean_sq_diff)
  return(rmse_value)
}

attr(rmse, "py_function_name") <- "rmse"

# Model compilation and callbacks------------------------------------------
model_0 %<>% compile(
  optimizer = "adam",
  loss="mse", 
  metrics = list("mae",
                 "mse",
                 rmse
  )
)

# Secuencia del modelo:

number_sequence <- 0
file_name <- paste0("lstm_QC_", number_sequence, ".keras") 

callbacks <- list(
  callback_tensorboard(log_dir=file.path(resultsdir, "run_a")
                       # write_grads = T,
                       # histogram_freq = 1,
                       # update_freq = "batch",
                       # write_images = T
  ),
  callback_model_checkpoint(file.path(resultsdir, file_name),
                            monitor= "rmse",
                            mode="min",
                            verbose=1,
                            save_best_only = TRUE),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                patience = 10, min_lr = 0.000001),
  callback_early_stopping(monitor = "val_loss", patience = 20,
                          verbose = 1)
)


# Train model--------------------------------------------------------------

history_0 <- model_0 %>% keras::fit(
  train_k,
  steps_per_epoch = train_steps,
  epochs = 15,
  validation_data = val_k,
  validation_steps = val_steps,
  callbacks = callbacks
)

# Evaluate model-----------------------------------------------------------

evaluation_result <- evaluate(model_0, test_k, steps = test_steps)

evaluation_result

