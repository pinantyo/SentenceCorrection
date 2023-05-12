import tensorflow as tf

"""
	Model Architecture
"""
class Bi_LSTM_Seq2Seq(tf.keras.Model):
	def __init__(self, original, target, units=512):
	    super().__init__()
	    
	    self.enc_process_text = original
	    self.dec_process_text = target

	    self.enc_vocab_size = len(original.get_vocabulary())
	    self.dec_vocab_size = len(target.get_vocabulary())


	    # Encoder Embeddings
	    self.enc_embedding = tf.keras.layers.Embedding(
	        self.enc_vocab_size,
	        output_dim=units,
	        mask_zero=True
	    )

	    self.encoder = tf.keras.layers.Bidirectional(
	        layer = tf.keras.layers.LSTM(
	            int(units/2),
	            return_sequences=True,
	            return_state=True
	        ),
	    )


	    # Decoder Embeddings
	    self.dec_embedding = tf.keras.layers.Embedding(
	        self.dec_vocab_size,
	        output_dim=units,
	        mask_zero=True
	    )

	    self.decoder = tf.keras.layers.LSTM(
	        int(units),
	        return_sequences=True,
	        return_state=True
	    )

	    # Attention
	    self.attention = tf.keras.layers.Attention()

	    # Output
	    self.out = tf.keras.layers.Dense(self.dec_vocab_size)
	  

	 def call(self, original, target):
	    # Encoder
	    enc_tokens = self.original_process_text(original)
	    enc_vectors = self.ori_embedding(enc_tokens)

	    enc_model, forward_h, forward_c, backward_h, backward_c = self.encoder(
	        self.encoder(enc_vectors)
	    )
	    
	    enc_state_h = tf.concat([forward_h, backward_h], -1)
	    enc_state_c = tf.concat([forward_c, backward_c], -1)

	    # Decoder
	    dec_tokens = self.target_process_text(target)
	    expected = dec_tokens[:, 1:]

	    teacher_forcing = dec_tokens[:, :-1]
	    dec_vectors = self.dec_embedding(teacher_forcing)


	    dec_model_attention = self.attention(
	        inputs=[
	            dec_vectors,
	            enc_model
	        ],
	        mask=[
	            dec_vectors._keras_mask,
	            enc_model._keras_mask
	        ]
	    )



	    transfer_vec, _, _ = self.decoder(
	        self.decoder(
	          dec_model_attention, 
	          initial_state=[
	              enc_state_h,
	              enc_state_c
	          ]
	        )
	    )

	    output = self.out(transfer_vec)

	    return output, expected, output._keras_mask


	def train(epochs, model, batch=64, shuffle=1000, test_size=0.2):

		# Init loss - metrics - optimizers
		loss_function_train = tf.keras.losses.SparseCategoricalCrossentropy(
		    from_logits=True,
		    reduction=tf.keras.losses.Reduction.NONE
		)

	  	loss_function_val = tf.keras.losses.SparseCategoricalCrossentropy(
	      	from_logits=True,
	      	reduction=tf.keras.losses.Reduction.NONE
	  	)

		metric_function_train = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
		metric_function_val = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

		optimizer = tf.keras.optimizers.RMSprop(
		    learning_rate=1e-3
		)

	  	train_losses = []
	  	train_accuracy = []
	  	val_losses = []
	  	val_accuracy = []

  		# Split Dataset
  		mid = int(datasets.cardinality().numpy() * (1 - test_size))

  		train_ds = datasets.take(mid)
  		test_ds = datasets.skip(mid)
  
  		# Shuffle Dataset
  		train_ds = train_ds.shuffle(shuffle).batch(batch).cache()
  		test_ds = datasets.shuffle(shuffle).batch(8).cache()

	  	for epoch in range(epochs):
	    	epoch_losses_train = []
	    	epoch_losses_val = []

	    	epoch_acc_train = []
	    	epoch_acc_val = []


	    # Training Step
	    for step, ((original, target), (original_val, target_val)) in enumerate(zip(train_ds, test_ds)):

	      	# Training
		    with tf.GradientTape() as tape:
		        # Foward-pass
		        logits, expected, mask = model(original, target, training=True)

		        # Compute loss
		        loss = loss_function_train(expected, logits)

		        # loss = tf.ragged.boolean_mask(loss, mask)
		        # loss = tf.reduce_sum(loss) * (1. / batch)

		        mask = tf.cast(expected != 0, loss.dtype)
		        loss *= mask
		        loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
		      
		    epoch_losses_train.append(loss.numpy())

		    # Retrieve loss
		    grads = tape.gradient(loss, model.trainable_weights)

		    # Update weight gradient-based
		    optimizer.apply_gradients(zip(grads, model.trainable_weights))


		    # Compute acc
		    logits = tf.argmax(logits, axis=-1)
		    logits = tf.cast(logits, expected.dtype)

		    match = tf.cast(logits == expected, tf.float32)
		    mask = tf.cast(logits != 0, tf.float32)


		    metric = tf.reduce_sum(match)/tf.reduce_sum(mask)

		    epoch_acc_train.append(
		        metric.numpy()
		    )

		    # metric_function_train.update_state(expected, logits)



		    # Validation
		    logits, expected, mask = model(original_val, target_val)


		    # Compute loss

		    loss = loss_function_val(expected, logits)

		    # loss = tf.ragged.boolean_mask(loss, val_mask)
		    # loss = tf.reduce_sum(loss) * (1. / 8)

		    mask = tf.cast(expected != 0, loss.dtype)
		    loss *= mask
		    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

		      
		    epoch_losses_val.append(loss.numpy())


		    # Compute acc
		    logits = tf.argmax(logits, axis=-1)
		    logits = tf.cast(logits, expected.dtype)

		    match = tf.cast(logits == expected, tf.float32)
		    mask = tf.cast(logits != 0, tf.float32)

		    acc = tf.reduce_sum(match)/tf.reduce_sum(mask)

		    epoch_acc_val.append(
		        acc.numpy()
		    )

		    # metric_function_val.update_state(val_expected, val_logits)
		    
		    if step % 2000 == 0:
		    	print(f"Step-{step} - loss: {epoch_losses_train[step]} acc: {epoch_acc_train[step]}")


		    steps_losses = np.mean(epoch_losses_train)
		    steps_acc = np.mean(epoch_acc_train)



		    train_losses.append(steps_losses)
		    train_accuracy.append(steps_acc)

		    val_losses.append(np.mean(epoch_losses_val))
		    val_accuracy.append(np.mean(epoch_acc_val))

		    # Reset metrics
		    # metric_function_train.reset_states()
		    # metric_function_val.reset_states()

		    print('Trained epoch: {}; loss: {}; accuracy: {}'.format(epoch, steps_losses, steps_acc))

		  
		wandb.log({
		    "train_loss": train_losses.numpy(),
		    "train_accuracy": train_accuracy.numpy(),
		    "val_loss": val_losses.numpy(),
		    "val_accuracy": val_accuracy.numpy()
		})
		  
		plt.plot(train_losses)
		plt.xlabel('Epochs')
		plt.ylabel('Losses')


		tf.saved_model.save(model, 'NMTForStyleTextTransfer')



"""
	LLM - T5
"""