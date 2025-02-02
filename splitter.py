import pandas as pd

def split_iris_extended(input_file, train_file, stream_file, train_ratio=0.8, random_state=42):
    """
    Splits the iris_extended dataset into two files:
      - train_file: used for offline model training
      - stream_file: used for streaming simulation
       
    Parameters:
      input_file (str): Path to the original iris_extended CSV file.
      train_file (str): Path to save the training portion.
      stream_file (str): Path to save the streaming portion.
      train_ratio (float): Proportion of data to use for training (default: 0.7).
      random_state (int): Random seed for reproducibility.
    """
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Shuffle the dataset to ensure randomness
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Determine the number of training samples
    split_index = int(len(df_shuffled) * train_ratio)
    
    # Split the data
    df_train = df_shuffled.iloc[:split_index]
    df_stream = df_shuffled.iloc[split_index:]
    
    # Save the resulting splits to CSV files
    df_train.to_csv(train_file, index=False)
    df_stream.to_csv(stream_file, index=False)
    
    print(f"Split completed: {len(df_train)} records for training and {len(df_stream)} records for streaming.")

# Example usage:
if __name__ == "__main__":
    split_iris_extended("iris_extended.csv", "iris_extended_train.csv", "iris_extended_stream.csv")
