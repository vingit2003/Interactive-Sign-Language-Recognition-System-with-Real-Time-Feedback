# Interactive Sign Language Recognition System with Real Time Feedback
This sign language recognition system detects six basic gestures: hello, thanks, welcome, sorry, yes, and strong. 
Using a camera feed, the model captures, processes, and classifies these gestures. Real-time feedback with confidence score and messages helps users adjust their gestures if needed.

**Tools and Technologies**
- MediaPipe: For landmark extraction, using the Holistic model.
- OpenCV: To capture video input and display feedback messages.
- TensorFlow/Keras: For building and training the LSTM model.
- Matplotlib: For data visualization.
- NumPy: For managing landmark data.
- Sklearn: For metrics and splitting dataset.

**Architecture of the model used**
  inputs = Input(shape=(num_frames_per_video, X.shape[2]))
    
  x1 = LSTM(128, return_sequences=True, activation='tanh')(inputs)
  x1 = Dropout(0.3)(x1)
  x1 = LayerNormalization()(x1)
    
  attention_output = MultiHeadAttention(
        num_heads=8, key_dim=32
    )(x1, x1, x1)
  attention_output = Dropout(0.2)(attention_output)
  x2 = LayerNormalization()(attention_output + x1)
    
  x3 = LSTM(256, return_sequences=True, activation='tanh')(x2)
  x3 = Dropout(0.3)(x3)
  x3 = LayerNormalization()(x3)
    
  attention_output_2 = MultiHeadAttention(
        num_heads=8, key_dim=32
    )(x3, x3, x3)
  attention_output_2 = Dropout(0.2)(attention_output_2)
  x4 = LayerNormalization()(attention_output_2 + x3)
    
  x5 = LSTM(128, return_sequences=True, activation='tanh')(x4)
  x5 = LayerNormalization()(x5)
    
  x6 = GlobalAveragePooling1D()(x5)
    
  x7 = Dense(256, activation='relu')(x6)
  x7 = Dropout(0.4)(x7)
  x8 = Dense(128, activation='relu')(x7)
  x8 = Dropout(0.3)(x8)
    
  outputs = Dense(actions.shape[0], activation='softmax')(x8)
    
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
  )
