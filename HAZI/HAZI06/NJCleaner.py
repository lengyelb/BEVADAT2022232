import pandas as pd


class NJCleaner:
    def __init__(self, csv_path: str) -> None:
        self.data = pd.read_csv(csv_path)

    def order_by_scheduled_time(self) -> pd.DataFrame:
        return self.data.sort_values(by=['scheduled_time'])

    def drop_columns_and_nan(self):
        return self.data.drop(['from', 'to'], axis=1).dropna()

    def convert_date_to_day(self):
        days_df = self.data.copy()
        days_df['day'] = pd.to_datetime(days_df['date']).dt.day_name()
        return days_df.drop('date', axis=1)

    def convert_scheduled_time_to_part_of_the_day(self):
        def get_part_of_the_day(time):
            hour = int(time.split(' ')[1].split(':')[0])
            if 4 <= hour <= 7:
                return 'early_morning'
            elif 8 <= hour <= 11:
                return 'morning'
            elif 12 <= hour <= 15:
                return 'afternoon'
            elif 16 <= hour <= 19:
                return 'evening'
            elif 20 <= hour <= 23:
                return 'night'
            else:
                return 'late_night'

        part_of_day_df = self.data.copy()
        part_of_day_df['part_of_the_day'] = part_of_day_df['scheduled_time'].apply(get_part_of_the_day)
        return part_of_day_df.drop('scheduled_time', axis=1)

    def convert_delay(self):
        delay_df = self.data.copy()
        delay_df['delay'] = (delay_df['delay_minutes'] >= 5).astype(int)
        return delay_df

    def drop_unnecessary_columns(self):
        return self.data.drop(['train_id', 'actual_time', 'delay_minutes'], axis=1)

    def save_first_60k(self, path):
        self.data.iloc[:60000].to_csv(path, index=False)

    def prep_df(self, path='data/NJ.csv'):
        self.data = self.order_by_scheduled_time()
        self.data = self.drop_columns_and_nan()
        self.data = self.convert_date_to_day()
        self.data = self.convert_scheduled_time_to_part_of_the_day()
        self.data = self.convert_delay()
        self.data = self.drop_unnecessary_columns()
        self.save_first_60k(path)
