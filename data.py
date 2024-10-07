import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

#preproces the df for dataloader

def preprocess(df,take_sample=False,sample=5000):
    # Normalize the Amount column
    df['Amount_normalized'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

    # Drop the Time column
    df = df.drop(columns=['Time'])
    #Drop Amount column
    df = df.drop(columns=['Amount'])
    if take_sample:
        if sample > df[df['Class'] == 0].shape[0]:
            raise ValueError('Sample size is greater than the number of samples where Class == 0')
        # Take a random sample of sample where Class == 0
        df_class_0 = df[df['Class'] == 0].sample(sample, random_state=42)
        df_class_0 = df_class_0.drop(columns=['Class'])
    else:
        # Filter a new DataFrame that contains only Class == 0
        df_class_0 = df[df['Class'] == 0]
        df_class_0 = df_class_0.drop(columns=['Class'])
    # Filter a new DataFrame that contains only Class == 1
    df_class_1 = df[df['Class'] == 1]
    df_class_1 = df_class_1.drop(columns=['Class'])

    #create a validaton set from df_class_0
    df_class_0_train = df_class_0.sample(frac=0.8, random_state=42)
    #create a vali and test set from rest of class 0
    df_class_0_val = df_class_0.drop(df_class_0_train.index).sample(frac=0.5, random_state=42)
    df_class_0_test = df_class_0.drop(df_class_0_train.index).drop(df_class_0_val.index)

    return df_class_0_train,df_class_0_val,df_class_0_test,df_class_1


class CustomDataset(Dataset):
    def __init__(self, df):
        """
        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
        """
        # Convert the DataFrame to a PyTorch tensor
        self.data = torch.tensor(df.values, dtype=torch.float32)

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample of data.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            torch.Tensor: The data sample at the specified index.
        """
        return self.data[idx]
    


if __name__ == '__main__':
    run_example=False
    if run_example:
        df=pd.read_csv('creditcard.csv')
        df_class_0_train,df_class_0_val,df_class_0_test,df_class_1=preprocess(df,take_sample=True,sample=5000)

        print(df_class_0_train.head())
        print(df_class_0_val.head())
        print(df_class_0_test.head())
        print(df_class_1.head())


        train_dataset = CustomDataset(df_class_0_train)
        val_dataset = CustomDataset(df_class_0_val)
        test_dataset_class_0 = CustomDataset(df_class_0_test)
        test_dataset_class_1 = CustomDataset(df_class_1)
