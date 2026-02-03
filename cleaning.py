import pandas as pd
import numpy as np
import os 

# Membuat class agar bisa di import di file lain
class Weathercleaner:
    # Membuat fungsi utama untuk menyimpan file
    def  __init__(self,file: str):
        self.file = file
        self.df = None
    # Membuat fungsi untuk membaca dataset
    def load_data(self):
        # Membuat Error Handling jika file tidak ditemukan
        if not os.path.exists(self.file):
            raise FileNotFoundError("File Not Found")
        self.df = pd.read_csv(self.file)
        return self.df
    # Membuat fungsi untuk menghapus kolom yang memiliki banyak missing value (35%+)
    def drop_data(self):
        self.df.drop(['Sunshine','Evaporation','Cloud3pm','Cloud9am'],axis = 1,inplace=True)
        self.df.dropna(subset=['RainTomorrow'], inplace=True)
        return self.df
    # Membuat fungsi untuk mengambil bulan di kolom date
    def retrieve_data(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.month
        self.df.drop('Date', axis=1, inplace=True)
        return self.df
    # Membuat fungsi untuk mengisi nilai missing
    def fill_missing(self):
        numeric = self.df.select_dtypes(include=np.number).columns
        category = self.df.select_dtypes(exclude=np.number).columns

        for col in numeric:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        for col in category:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
    
    def replace_location(self):
        self.df['Location'] = self.df["Location"].str.replace("Airport","")

    # Membuat fungsi untuk menjalankan fungsi lain
    def run_cleaner(self):
        self.load_data()
        self.drop_data()
        self.retrieve_data()
        self.fill_missing()
        self.replace_location()
        return self.df

# Untuk mencegah agar ketika di import tidak ke run otomatis
if __name__ == "__main__":
    cleaner = Weathercleaner("dataset/raw_weatherAUS.csv")
    df_clean = cleaner.run_cleaner()
    # Menyimpan data yang sudah bersih
    df_clean.to_csv("dataset/weather_cleaned.csv", index=False)


