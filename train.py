import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle
import json
import glob
import os

class AttackPredictor:
    def __init__(self, max_words=15000, max_len=250):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.technique_encoder = LabelEncoder()
        self.tactic_encoder = LabelEncoder()
        self.model = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess all test data files"""
        # Search for data files in Test_Data directory
        data_dir = os.path.join(os.getcwd(), 'Test_Data')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Test_Data directory not found at {data_dir}")
        
        # Load all CSV files from the Test_Data directory
        all_files = glob.glob(os.path.join(data_dir, 'test_data_*.csv'))
        if not all_files:
            raise ValueError(f"No test data files found in {data_dir}")
            
        print(f"Found {len(all_files)} data files")
        dataframes = []
        
        for file in all_files:
            try:
                df = pd.read_csv(file)
                # Create description from available fields
                df['description'] = df.apply(lambda row: f"Incident {row['incident_id']} of {row['severity']} severity involved "
                                                       f"primary tactic {row['primary_tactic']} using technique {row['technique_id']}. "
                                                       f"Secondary tactics observed: {row['secondary_tactics']} "
                                                       f"with techniques: {row['secondary_techniques']}", axis=1)
                dataframes.append(df)
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
        
        if not dataframes:
            raise ValueError("No valid data files could be loaded")
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total records loaded: {len(combined_df)}")
        return combined_df
    
    def preprocess_data(self, df):
        """Preprocess the text data and encode labels"""
        print("Processing descriptions...")
        # Tokenize descriptions with better text preprocessing
        descriptions = df['description'].str.lower()  # Convert to lowercase
        self.tokenizer.fit_on_texts(descriptions)
        X = self.tokenizer.texts_to_sequences(descriptions)
        X = pad_sequences(X, maxlen=self.max_len, padding='post', truncating='post')
        
        print("Processing techniques and tactics...")
        # Process techniques and tactics
        techniques = df['technique_id']
        y_techniques = self.technique_encoder.fit_transform(techniques)
        
        tactics = df['primary_tactic']
        y_tactics = self.tactic_encoder.fit_transform(tactics)
        
        print(f"Preprocessed {len(X)} samples")
        return X, y_techniques, y_tactics
    
    def build_model(self, vocab_size, n_techniques, n_tactics):
        """Build and compile an improved LSTM model"""
        model = Sequential([
            Embedding(vocab_size, 200, input_length=self.max_len),
            Dropout(0.2),
            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(128)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(n_techniques, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X, y_techniques, y_tactics, epochs=4, batch_size=32):
        """Train the model with improved training process"""
        X_train, X_val, y_train, y_val = train_test_split(X, y_techniques, 
                                                         test_size=0.1,
                                                         random_state=42,
                                                         stratify=y_techniques)
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        early_stopping = EarlyStopping(monitor='val_accuracy',
                                     patience=5,
                                     restore_best_weights=True,
                                     mode='max')
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.2,
                                     patience=3,
                                     min_lr=0.00001)
        
        history = self.model.fit(X_train, y_train,
                               validation_data=(X_val, y_val),
                               epochs=epochs,
                               batch_size=batch_size,
                               callbacks=[early_stopping, reduce_lr],
                               class_weight=self._compute_class_weights(y_train))
        
        return history
    
    def _compute_class_weights(self, y_train):
        """Compute class weights to handle imbalanced data"""
        classes = np.unique(y_train)
        weights = dict(zip(classes, 
                         [len(y_train) / (len(classes) * np.sum(y_train == c)) for c in classes]))
        return weights
    
    def save_model(self, model_path='attack_model.h5', 
                  tokenizer_path='tokenizer.pkl',
                  encoders_path='encoders.pkl'):
        """Save the model and preprocessing objects"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to {tokenizer_path}")
            
        encoders = {
            'technique_encoder': self.technique_encoder,
            'tactic_encoder': self.tactic_encoder
        }
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        print(f"Encoders saved to {encoders_path}")

def main():
    try:
        # Initialize predictor
        print("Initializing predictor...")
        predictor = AttackPredictor()
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = predictor.load_and_preprocess_data()
        X, y_techniques, y_tactics = predictor.preprocess_data(df)
        
        # Build model
        print("Building model...")
        vocab_size = len(predictor.tokenizer.word_index) + 1
        n_techniques = len(predictor.technique_encoder.classes_)
        n_tactics = len(predictor.tactic_encoder.classes_)
        print(f"Vocabulary size: {vocab_size}")
        print(f"Number of techniques: {n_techniques}")
        print(f"Number of tactics: {n_tactics}")
        predictor.build_model(vocab_size, n_techniques, n_tactics)
        
        # Train model
        print("Training model...")
        history = predictor.train(X, y_techniques, y_tactics)
        
        # Save model and preprocessing objects
        print("Saving model...")
        predictor.save_model()
        
        # Print final metrics
        final_accuracy = history.history['accuracy'][-1]
        final_val_accuracy = history.history['val_accuracy'][-1]
        print(f"\nTraining completed!")
        print(f"Final training accuracy: {final_accuracy:.4f}")
        print(f"Final validation accuracy: {final_val_accuracy:.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()