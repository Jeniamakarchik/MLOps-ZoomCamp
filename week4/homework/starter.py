import argparse
import pickle
from pathlib import Path

import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(
                prog='Predictor',
                description='Model predicts duration of the taxi trip'
            )
    
    parser.add_argument('-y', '--year', type=int)
    parser.add_argument('-m', '--month', type=int)
    parser.add_argument('-t', '--taxi_type', type=str)

    return parser.parse_args()


def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    args = parse_args()

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{args.taxi_type}_tripdata_{args.year:04d}-{args.month:02d}.parquet'
    output_file = Path(f'output/{args.taxi_type}/{args.year:04d}-{args.month:02d}.parquet')

    print(f'Reading thr data from {url}')
    df = read_data(url)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    print('Making predictions...')
    y_pred = model.predict(X_val)

    print(f'Mean: {y_pred.mean():.2f}')
    print(f'Standard deviation: {y_pred.std():.2f}')

    print('Saving results...')
    df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')
    save_results(df, y_pred, output_file)


if __name__ == '__main__':
    run()
