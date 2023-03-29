from ast import literal_eval

from tqdm import trange


def get_data(df):
    df["spans"] = df.spans.apply(literal_eval)
    for i in trange(len(df)):
        text = df['text'][i]
        spans = df['spans'][i]
        label = [0 for _ in range(len(text))]
        for toxic_position in spans:
            label[toxic_position] = 1
        df['spans'][i] = label
    return df